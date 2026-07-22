using System.Net;
using System.Net.Http;
using System.Net.Http.Json;
using Microsoft.Web.WebView2.Core;

namespace Custback.Shell;

/// <summary>
/// Establishes the HttpOnly browser session for the WebView without ever
/// placing the long-lived bearer into a URL, command line, web storage, or log
/// (WIN-5.3).
///
/// The shell exchanges the bearer for an opaque session cookie over loopback
/// itself and injects only that session cookie into WebView2. The page then
/// loads already-authenticated at the loopback origin and never sees, stores,
/// or transmits the bearer. The bearer lives only in the engine's private token
/// file and this process's memory.
/// </summary>
internal static class SessionBootstrap
{
    private const string SessionCookieName = "custback_session";

    internal static async Task InjectSessionAsync(CoreWebView2 webView, Engine engine)
    {
        string session = await MintSessionAsync(engine).ConfigureAwait(true);

        var cookie = webView.CookieManager.CreateCookie(SessionCookieName, session, "127.0.0.1", "/");
        cookie.IsHttpOnly = true;
        cookie.SameSite = CoreWebView2CookieSameSiteKind.Strict;
        // Loopback is plain http; the engine only marks the cookie Secure under
        // TLS. Matching that here keeps the cookie from being dropped.
        cookie.IsSecure = false;
        webView.CookieManager.AddOrUpdateCookie(cookie);
    }

    private static async Task<string> MintSessionAsync(Engine engine)
    {
        var handler = new SocketsHttpHandler { UseProxy = false, AllowAutoRedirect = false };
        using var http = new HttpClient(handler)
        {
            BaseAddress = new Uri(engine.BaseUrl),
            Timeout = TimeSpan.FromSeconds(5),
        };

        // POST body carries the token — never a query string. Host is
        // 127.0.0.1:<port> (from BaseAddress) with no Origin, which the engine's
        // loopback security boundary accepts for a native client.
        using var request = new HttpRequestMessage(HttpMethod.Post, "/auth/session")
        {
            Content = JsonContent.Create(new { token = engine.Bearer }),
        };
        using var response = await http.SendAsync(request).ConfigureAwait(false);
        if (response.StatusCode != HttpStatusCode.NoContent)
        {
            throw new InvalidOperationException(
                $"session bootstrap rejected by engine: HTTP {(int)response.StatusCode}");
        }

        if (!response.Headers.TryGetValues("Set-Cookie", out var cookies))
        {
            throw new InvalidOperationException("session bootstrap returned no Set-Cookie");
        }

        foreach (var value in cookies)
        {
            // custback_session=<opaque>; HttpOnly; SameSite=Strict; Path=/
            const string marker = SessionCookieName + "=";
            int start = value.IndexOf(marker, StringComparison.Ordinal);
            if (start < 0)
            {
                continue;
            }

            start += marker.Length;
            int end = value.IndexOf(';', start);
            return end < 0 ? value[start..] : value[start..end];
        }

        throw new InvalidOperationException("session bootstrap Set-Cookie had no session value");
    }
}
