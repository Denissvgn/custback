---
name: verify
description: Build/launch/drive recipe to verify custback changes at their runtime surfaces (avatar service, control APIs).
---

# Verifying custback changes

Interpreter: `.venv/bin/python` (plain `python` is not on PATH; `pytest` on
PATH is the system one — always go through the venv).

## Avatar service (stage 2) end-to-end

No camera or vcam device is needed: stand in for custback with a loopback
`websockets` server that serves `/ws/frames?stream=raw`, checks
`Authorization: Bearer <token>`, streams synthetic camera JPEGs (~15 fps),
and asserts returned frames are JPEGs of exactly the camera size (that is
custback's hard contract). See `tests/test_avatar_service.py::FakeCustback`
for the shape; a standalone copy works as a background process.

1. Create mode-0600 token files in a scratch dir: one for the source
   connection, one for the avatar control API.
2. Start the stand-in on a free port (e.g. 18710), then the real service:

   ```bash
   .venv/bin/python -m custback.avatar \
     --source ws://127.0.0.1:18710 --source-token-file $SCRATCH/api-token \
     --driver idle --api-port 18711 --api-token-file $SCRATCH/avatar-api-token -v
   ```

   `--driver idle` keeps it deterministic (no mediapipe needed).
3. Drive the control API with `curl -H "Authorization: Bearer $(cat
   $SCRATCH/avatar-api-token)"`: `GET /status` (connected/frames_sent/
   render_failures), `GET /avatars`, `PATCH /config` for hot appearance/
   background changes, `GET /video/snapshot.jpg` to capture rendered
   frames.
4. Visual checks: save snapshots per config combo, build a labeled contact
   sheet with cv2 (`hstack`/`vstack` + `putText`), and Read the image.

Gotchas
- The service log with `-v` is very chatty (per-frame websockets DEBUG
  lines); grep rather than tail.
- After a PATCH, sleep ~0.5 s before snapshotting — hot config applies on
  the next rendered frame.
- Bad hot patches must never stop the stream: after probing invalid
  PATCHes, confirm `frames_sent` still climbs and `render_failures` is 0.

## Release gates

`node scripts/release/verify-release.js` and `npm test` (from the repo
root; `packaging/npm` has no package.json of its own). Both pin exact file
manifests — adding any packaged file requires updating them.
