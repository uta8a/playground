# try-devtools-performance-api

Honoで画像付きのHTMLを配信する、ローカル用の最小サンプルです。
HTMLのimgタグから `sample.png` を読み込み、インラインJavaScriptから
`fetch('/foo')` でJSON APIを呼び出します。

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

## fetchとperformance.mark

ページを開くたびに `GET /foo` を1回呼び出します。Honoは
`{ "message": "Hello from /foo" }` をJSONで返し、結果をConsoleに表示します。

`performance.mark('hoge')` は `hoge` という名前で時点を記録します。
名前は文字列なので引用符を付けます。`startTime` はページの時間原点からの
経過時間（ミリ秒）で、マーク自体の `duration` は0です。

HTML内では次の順序で計測します。

1. `performance.mark('hoge')` でfetch開始直前を記録
2. `fetch('/foo')` と `response.json()` で本文の受信・JSON解析を完了
3. `performance.mark('foo-end')` で終了時点を記録
4. `performance.measure('foo-fetch', 'hoge', 'foo-end')` で2点間の経過時間を記録

`foo-fetch` はサーバーの処理時間だけではなく、通信とJSON解析などを含む時間です。
リクエストに失敗した場合はConsoleにエラーを表示し、成功時の終了マークとmeasureは作成しません。

Consoleでマーク・計測結果・fetchのResource Timingを確認できます。

```js
performance.getEntriesByType('mark').map((entry) => entry.toJSON())
performance.getEntriesByName('foo-fetch', 'measure').map((entry) => entry.toJSON())
performance.getEntriesByType('resource')
  .filter((entry) => entry.initiatorType === 'fetch')
  .map((entry) => entry.toJSON())
```

## 参考

- [HonoのNode.js向けドキュメント](https://hono.dev/docs/getting-started/nodejs)
- [MDN: performance.mark()](https://developer.mozilla.org/en-US/docs/Web/API/Performance/mark)
- [MDN: performance.measure()](https://developer.mozilla.org/en-US/docs/Web/API/Performance/measure)

---

# 試したログ

`fetch` と `performance.mark()` も試してみた

```
performance.mark("A")
// 処理
performance.mark("B")
const c = performance.measure("measure-name", "A", "B")

console.log(c.duration)
```

開始・終了みたいなのだけでなく、markを打っておいてmark間の差分を出すみたいなことができるっぽい。なるほどね。
あと、performanceはglobal objectなので、tryの前に打ってtry内で終了をmarkするということが可能。

APIもresource typeに当たる。markはmark typeに当たる。
