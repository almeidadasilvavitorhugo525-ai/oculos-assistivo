// Service worker que liga o "isolamento" (cross-origin isolation) da página.
// Sem isso o navegador só deixa a IA usar 1 núcleo do processador.
// O GitHub Pages não permite configurar esses cabeçalhos, então eles são
// acrescentados aqui, só na resposta da própria página.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.mode !== "navigate") return; // demais arquivos passam direto

  event.respondWith(
    fetch(req).then((res) => {
      if (!res.ok && res.status !== 304) return res;
      const headers = new Headers(res.headers);
      headers.set("Cross-Origin-Opener-Policy", "same-origin");
      // "credentialless" deixa carregar os scripts e modelos dos CDNs sem mudança neles.
      headers.set("Cross-Origin-Embedder-Policy", "credentialless");
      return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
    })
  );
});
