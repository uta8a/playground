# naoyaさんのHaskellのAtCoderテンプレから学ぶ

[abc402 a](https://atcoder.jp/contests/abc402/submissions/64997179) のnaoyaさんのコードを元に、Haskellのライブラリ整備について学ぶ。

- 構造は3つ
  - Imports and Language Extensions
  - Main: ここにコードを書く
  - My Prelude

全てMy Preludeに寄せられないかなと考えたが、Haskellのimportは後方から前方に参照できないっぽい(観察によるもので、ドキュメントは見つけられなかった) まあ普通importの前では使えないよね。

`{-- IO --}` のセクションに色々書かれている。

今回はString系を見ていきたい。

```haskell
getCharArray :: (Int, Int) -> IO (UArray Int Char)
getCharArray b = listArray @UArray b <$> getLine
```

getCharArrayの引数: `(lower, upper)` 文字列の長さの範囲指定

[Haskell の Array](https://zenn.dev/naoya_ito/articles/87a8a21d52c302) を読む。

メンタルモデル的に、配列の配列、じゃなくてboundの方に注目した方が良さそう。`Ix` の使いこなしに慣れるのも必要。

String系、よく考えたら `BS.getLine` で良さそう。
そのあとの処理をどうするかによって、 `UArray` にするかとかが決まる。

今はcharの配列が欲しい気持ちなので、 `UArray` にしてみる。
