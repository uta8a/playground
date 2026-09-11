# try-empty-html-performance-api

Honoで空のHTMLを配信する、ローカル用の最小サンプルです。
HTMLのbodyは空で、JavaScriptやCSSなどの追加リソースはありません。

## 起動

Node.js 22以降とnpmを使用します。

```sh
cd try-empty-html-performance-api
npm ci
npm start
```

ブラウザで http://127.0.0.1:3000 を開くと空白のページが表示されます。
サーバーは `127.0.0.1:3000` のみで待ち受けます。終了するには `Ctrl+C` を押します。
`index.html` を変更した場合はサーバーを再起動してください。

## Performance APIの確認

ページを読み込んだ後、ブラウザの開発者ツールのConsoleで実行できます。

```js
performance.getEntriesByType('navigation')[0].toJSON()
```

## 参考

- [HonoのNode.js向けドキュメント](https://hono.dev/docs/getting-started/nodejs)
