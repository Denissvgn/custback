# Two-host avatar deployment

This guide covers the supported split deployment: `custback` owns the camera
and virtual-camera output on the **meeting host**, while `custback-avatar`
renders on the **renderer host**. The hosts use two independent, TLS-protected
connections. Do not reuse a token, certificate, or private key between them.

## Trust planes and firewall direction

| Plane | Initiator | Listener | Transport | Credential | Firewall rule |
| --- | --- | --- | --- | --- | --- |
| Renderer frames | renderer host | meeting host, `api.port` (8710 by default) | `wss://` | renderer-scoped token from `api.renderer_token_file` | Allow renderer host **outbound to** meeting host TCP 8710; allow that inbound flow on the meeting host |
| Avatar control | meeting host | renderer host, avatar `api.port` (8711 by default) | `https://` | avatar-control token from the avatar service's `api.token_file` | Allow meeting host **outbound to** renderer host TCP 8711; allow that inbound flow on the renderer host |

The renderer token is accepted only by the raw-frame WebSocket. The distinct
avatar-control token authorizes the renderer host's REST control API and is
used by custback's `/avatar/*` proxy. Neither token is the meeting host's
management token (`api.token_file`), and none should be sent by a browser to
the renderer host. Restrict both firewall rules to the named peer addresses;
the renderer control port does not need to be reachable from the user network.

If operators access custback's web UI over the network, separately allow their
trusted network to the meeting host's HTTPS port. That browser connection uses
custback's management login and is not either host-to-host credential above.

## Provision credentials and certificates

Create two independent random tokens under a restrictive umask. Distribute
each value through the deployment secret manager to exactly the two paths
shown below; do not put tokens in YAML, process arguments, images, or source
control.

| Secret | Authoritative listener copy | Client copy |
| --- | --- | --- |
| Renderer token | meeting: `api.renderer_token_file` | renderer: `source.token_file` |
| Avatar-control token | renderer: avatar `api.token_file` | meeting: `avatar.token_file` |

Token directories should be mode `0700` and files mode `0600`. For example,
run `umask 077` before generating a 32-byte random value with the platform's
secret tooling, then install it atomically at both endpoints. The listener and
client copies must contain the same value, but the two rows must not.

Use certificates whose subject alternative names match the exact DNS names in
the client URLs:

- The meeting host presents its server certificate on `wss://meeting.example:8710`.
  The renderer trusts its private CA through `source.tls_ca_file`; leave that
  field empty only when the issuer is already in the system trust store.
- The renderer host presents a different server certificate on
  `https://renderer.example:8711`. The meeting host trusts its private CA
  through `avatar.tls_ca_file`.
- Keep every private key readable only by the service account. Configure each
  certificate and key as a pair. Never disable hostname or chain validation.
- `source.tls_certfile`/`source.tls_keyfile` and
  `avatar.tls_certfile`/`avatar.tls_keyfile` provide outbound client identities
  when a front proxy enforces mTLS. The built-in servers provide TLS plus
  Bearer authentication; terminate mTLS at a proxy configured with the client
  CA and bind its upstream to loopback.

## Configure the meeting host

Start from `config/default.yaml` and set these deployment-owned values:

```yaml
background:
  mode: remote

api:
  host: 0.0.0.0
  port: 8710
  allow_non_loopback: true
  allowed_origins: [https://meeting.example:8710]
  renderer_token_file: /etc/custback/renderer-token
  tls_certfile: /etc/custback/tls/meeting.crt
  tls_keyfile: /etc/custback/tls/meeting.key

avatar:
  url: https://renderer.example:8711
  token_file: /etc/custback/avatar-control-token
  tls_ca_file: /etc/custback/tls/renderer-ca.pem
  tls_certfile: ""  # set both client fields only when an mTLS proxy requires them
  tls_keyfile: ""
```

Run `custback -c /etc/custback/config.yaml`. The existing management token at
`api.token_file` remains separate from both host-to-host secrets.

## Configure the renderer host

Start from `config/avatar.yaml` and set the matching peer settings:

```yaml
source:
  url: wss://meeting.example:8710
  token_file: /etc/custback/renderer-token
  tls_ca_file: /etc/custback/tls/meeting-ca.pem
  tls_certfile: ""  # set both client fields only when an mTLS proxy requires them
  tls_keyfile: ""

api:
  host: 0.0.0.0
  port: 8711
  allow_non_loopback: true
  allowed_origins: [https://meeting.example:8710]
  token_file: /etc/custback/avatar-control-token
  tls_certfile: /etc/custback/tls/renderer.crt
  tls_keyfile: /etc/custback/tls/renderer.key
```

Run `custback avatar -c /etc/custback/avatar.yaml` (the installed
`custback-avatar` alias is equivalent). A remote `ws://` or `http://` URL is
rejected; plaintext is supported only on numeric loopback for same-host use.

## Rotation

Token files and outbound trust settings are startup authority, so rotation
requires service restarts. There is no unsafe dual-token grace period.

1. Schedule the brief affected-plane outage and generate a new value without
   replacing the other plane's token.
2. Atomically install mode-`0600` copies on the listener and client hosts.
3. Restart the listener that validates the token, then restart the client.
4. Verify the connection and an intentionally wrong-token rejection. Destroy
   old secret-manager versions according to the site's retention policy.

For renderer-token rotation, custback shows the privacy slate until the avatar
client reconnects. For avatar-control-token rotation, `/avatar/*` control calls
fail closed while frame rendering can continue. Restart custback after changing
its `avatar.token_file` copy because the proxy snapshots that credential.

Rotate a private CA without a trust gap: first deploy an old-plus-new CA bundle
to each client and restart it, then replace the listener certificate/key and
restart the listener, verify hostname and chain validation, and only then
remove the old CA from clients. Rotate an mTLS client identity in the analogous
order: make the proxy trust both issuers, replace the client pair, verify, then
retire the old issuer.

## Failure and recovery contract

- If WSS is unreachable, its certificate is invalid, or the renderer token is
  rejected, custback-avatar reconnects with bounded backoff. Custback emits its
  fixed opaque privacy slate after `api.remote_timeout_ms`; it never exposes a
  raw, stale, previous-session, or local-camera fallback frame.
- If the HTTPS avatar-control plane is unreachable, custback maps proxied
  control requests to `502 avatar_unreachable`. A rejected control token maps
  to `502 avatar_auth_failed`. The user's custback browser session is not
  invalidated, and the independent renderer-frame plane can continue.
- If either TLS chain or hostname cannot be verified, treat it as an outage.
  Do not bypass verification or fall back to plaintext while recovering.
- Restarting the renderer may interrupt both planes briefly. The meeting host
  remains fail-closed on the privacy slate until a fresh authenticated renderer
  session returns valid output.

Before production, test both nominal directions, remove each firewall rule in
turn, present an untrusted certificate on each listener, and try the wrong token
on each plane. Confirm the status codes and privacy-slate behavior above, then
restore the secrets without printing them in logs or shell history.
