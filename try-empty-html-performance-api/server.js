import { readFileSync } from 'node:fs';
import { serve } from '@hono/node-server';
import { Hono } from 'hono';

const html = readFileSync(new URL('./index.html', import.meta.url), 'utf8');
const app = new Hono();

app.get('/', (c) => c.html(html));

serve(
  { fetch: app.fetch, hostname: '127.0.0.1', port: 3000 },
  (info) => console.log(`Listening on http://127.0.0.1:${info.port}`),
);
