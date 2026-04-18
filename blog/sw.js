self.addEventListener("install", function () {
  self.skipWaiting();
});

self.addEventListener("activate", function (event) {
  event.waitUntil(
    self.registration.unregister().then(function () {
      return self.clients.matchAll();
    }).then(function (clients) {
      clients.forEach(function (client) {
        client.navigate(client.url);
      });
    })
  );
});
