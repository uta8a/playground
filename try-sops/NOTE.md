# sops オプションを試すコマンド例

## 前提

```bash
cd <path-to-playground>/try-sops
export SOPS_AGE_KEY_FILE=./key.txt # これが大事。復号時にageのsecret keyが必要。
```

## すぐ試せるコマンド

### 1) `decrypt` / `--decrypt`

```bash
sops -d --output-type json test.json.enc
```

### 2) `--extract`（特定キーだけ復号）

```bash
sops decrypt --extract '["fuga"]' test.json.enc
```

### 3) `set`（暗号化ファイルのキー更新）

`set` は対象ファイルを in-place 更新する。

```bash
cp test.json.enc /tmp/test.enc.set.json
sops set /tmp/test.enc.set.json '["fuga"]' '"changed-by-set"'
sops decrypt --extract '["fuga"]' /tmp/test.enc.set.json
```

### 4) `unset`（暗号化ファイルのキー削除）

`unset` も対象ファイルを in-place 更新する。

```bash
cp test.json.enc /tmp/test.enc.unset.json
sops unset /tmp/test.enc.unset.json '["hoge"]'
sops decrypt /tmp/test.enc.unset.json
```

### 5) `rotate`（データキー再生成）

```bash
cp test.json.enc /tmp/test.enc.rotate.json
sops rotate --in-place /tmp/test.enc.rotate.json
```

確認例（`lastmodified` が更新される）:

```bash
rg '"lastmodified"' /tmp/test.enc.rotate.json
```

### 6) `exec-env`（復号値を環境変数で渡して実行）

```bash
sops exec-env test.json.enc 'env | rg "^(hoge|fuga)="'
```

### 7) `exec-file`（復号内容を一時ファイルで渡して実行）

```bash
sops exec-file --output-type json test.json.enc 'cat {}'
```

### 8) `--decryption-order`（復号に使う鍵種の優先順）

```bash
SOPS_DECRYPTION_ORDER='age,pgp' sops decrypt test.json.enc
# または
sops --decryption-order 'age,pgp' decrypt test.json.enc
```

### 9) `--input-type` / `--output-type`（形式変換を伴う復号）

```bash
sops decrypt --input-type json --output-type yaml test.json.enc
```

### 10) `--encrypted-regex`（暗号化対象キーの制御）

```bash
cat > /tmp/regex-target.json <<'JSON'
{"public":"visible","secret":"hidden"}
JSON
sops encrypt --input-type json --output-type json --encrypted-regex '^(secret)$' --age 'age19470el849wcpxatql6jrklp8xn88nnmq6njgz8q3xr34qvn78q3sznfm5g' /tmp/regex-target.json
```

## 前提ファイルが必要なコマンド

### `updatekeys`

`updatekeys` は `.sops.yaml` などの設定ファイルに従って master key を更新する。

```bash
sops updatekeys test.json.enc
```

### `groups`

`groups` は SOPS ファイルの key groups を編集する。
`groups add/delete` の細かい操作は `sops groups --help` で確認:

```bash
sops groups --help
```

ここまで検証済み。

## --以下は未検証--

参照: https://raw.githubusercontent.com/getsops/sops/refs/heads/main/README.rst

### 11) `publish`（復号して別ストアへ配布）

`.sops.yaml` の `destination_rules` を使って、S3/GCS/Vault へ配布できる。

```bash
sops publish s3/app.yaml
sops publish --recursive s3/
```

### 12) `--filename-override`（stdin 運用で creation rule を当てる）

パイプ入力だとファイル名がないので、`.sops.yaml` のルールマッチや format 推論のために指定する。

```bash
echo 'foo: bar' | sops encrypt --filename-override path/filename.sops.yaml > encrypted-data
cat encrypted-data | sops decrypt --filename-override filename.yaml > decrypted-data
```

### 13) `keyservice`（鍵を持たない端末から復号）

リモート側の key service を使って data key の暗号化/復号だけを委譲できる（README では SSH トンネル推奨）。

```bash
sops keyservice
sops decrypt --keyservice unix:///tmp/sops.sock file.yaml
sops decrypt --enable-local-keyservice=false --keyservice unix:///tmp/sops.sock file.yaml
```

### 14) `exec-env --same-process` / `exec-file --no-fifo`

`exec-env --same-process` はシグナル伝播が必要なサーバ起動時に便利。  
`exec-file` はデフォルト FIFO（メモリ渡し）で、必要なときだけ `--no-fifo` で一時ファイル化。

```bash
sops exec-env --same-process out.json './server'
sops exec-file --no-fifo out.json 'cat {}'
```

### 15) Git diff で平文比較

`.gitattributes` + `git config` で `git diff` 時だけ自動復号して差分確認できる。

```bash
echo '*.yaml diff=sopsdiffer' > .gitattributes
git config diff.sopsdiffer.textconv "sops decrypt"
git diff
```

## 小ネタ（README の話として面白い点）

- `age` は README 上でも PGP より推奨。
- `age` は SSH 公開鍵（`ssh-ed25519` / `ssh-rsa`）を recipient にできる。
- YAML の `anchor` は SOPS では非対応（認証に使う path が動的に変わるため）。
- YAML/JSON のトップレベル配列は不可（`sops` メタデータを置けないため）。
