# try-devtools-performance-api

Honoで画像付きのHTMLを配信する、ローカル用の最小サンプルです。
HTMLのimgタグから `sample.png` を読み込みます。JavaScriptやCSSはありません。

## 起動

Node.js 22以降とnpmを使用します。

```sh
cd try-devtools-performance-api
npm ci
npm start
```

ブラウザで http://127.0.0.1:3000 を開くと富士山の画像が表示されます。
サーバーは `127.0.0.1:3000` のみで待ち受けます。終了するには `Ctrl+C` を押します。
`index.html` を変更した場合はサーバーを再起動してください。
`try-empty-html-performance-api` と同じポートを使うため、同時には起動しないでください。

## Performance APIの確認

ページを読み込んだ後、ブラウザの開発者ツールのConsoleで実行できます。

```js
performance.getEntriesByType('navigation')[0].toJSON()
```

画像の読み込みに関する計測値も確認できます。

```js
performance.getEntriesByType('resource')
  .filter((entry) => entry.initiatorType === 'img')
  .map((entry) => entry.toJSON())
```

## 参考

- [HonoのNode.js向けドキュメント](https://hono.dev/docs/getting-started/nodejs)
