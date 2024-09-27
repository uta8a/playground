# 記事を読んだやつを個別のMarkdownファイルに分割する

## ディレクトリ構造

```
out/ # 出力先ディレクトリ
    TITLE.md
data.md # 記事データ
```

```md:data.md
# 記事タイトル
https://example.com/article1
記事の感想
# 記事タイトル2
https://example.com/article2
記事の感想2
```

## 使い方

```sh
deno task run

deno test -A # test

deno task clean # clean `out/`
```
