// 外语私教的 Service Worker：让手机能「安装」成 App（Android TWA 也要它）。
// 上课全靠实时连接，所以不缓存页面和接口；只在断网时给一个提示页。
const OFFLINE = `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>外语私教</title><body style="margin:0;font:16px -apple-system,'PingFang SC',sans-serif;background:#1f2a6b;color:#fff;
display:flex;align-items:center;justify-content:center;height:100vh;text-align:center;padding:24px">
<div><div style="font-size:48px">📶</div><h2>网络断开了</h2><p style="opacity:.8">连上网络后再打开外语私教就可以继续上课。</p>
<button onclick="location.reload()" style="font:inherit;border:0;border-radius:12px;padding:12px 22px;background:#fff;color:#1f2a6b">重试</button></div>`;

self.addEventListener("install", e => self.skipWaiting());
self.addEventListener("activate", e => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", e => {
  if (e.request.mode !== "navigate") return;   // 只管打开页面这一下；其余请求照常走网络
  e.respondWith(fetch(e.request).catch(() => new Response(OFFLINE, { headers: { "Content-Type": "text/html; charset=utf-8" } })));
});
