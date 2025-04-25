package main

import (
  "context"
  "os"
  "log/slog"
)

type ctxKey string

const TraceIDKey ctxKey = "traceID"

func main() {
  // 1) ロガー準備
  logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
  slog.SetDefault(logger)

  // 2) Context にトレース ID を埋め込む
  ctx := context.WithValue(context.Background(), TraceIDKey, "abc123")

  // 3) InfoContext で明示的に属性として渡す
  slog.InfoContext(ctx, "ユーザー認証成功",
    "userID", 1001,
    "traceID", ctx.Value(TraceIDKey),
  )
}
