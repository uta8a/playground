import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { serve } from '@hono/node-server';
import { serveStatic } from '@hono/node-server/serve-static';
import { Hono } from 'hono';

const html = readFileSync(new URL('./index.html', import.meta.url), 'utf8');
const app = new Hono();

app.get('/', (c) => c.html(html));
app.get('/foo', (c) => c.json({ message: 'Hello from /foo' }));
app.get('/sample.png', serveStatic({
  path: fileURLToPath(new URL('./sample.png', import.meta.url)),
}));

serve(
  { fetch: app.fetch, hostname: '127.0.0.1', port: 3000 },
  (info) => console.log(`Listening on http://127.0.0.1:${info.port}`),
);
