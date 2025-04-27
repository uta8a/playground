# Explore Rust

rust-lang/rust リポジトリを見てみる

496145b9cc023aef4bb1f16c0964a53d0da36c88 時点でのlocを調べる

```bash
tokei --output json > tokei.json
jq -r 'to_entries[] | .value.reports[] | select(.stats.code>=100) | select(.stats.code<=500) | select(.name | endswith(".rs")) | (.stats.code|tostring) + " " + .name\n' tokei.json > tokei.txt
cat tokei.txt| grep '\./compiler/' > tokei2.txt
sort -n -k1 tokei2.txt > tokei3.txt
```

100行以上500行以下のLOCのファイルを抽出したところ、`src/gcc/libgrust` や `./src/tools` 以下のファイルが多くヒットしたので、 `compiler` 以下に絞りました。

[loc-100-to-500.txt](loc-100-to-500.txt) を参照してください。
