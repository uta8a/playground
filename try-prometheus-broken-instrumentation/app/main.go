package main

import (
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// 0/1で状態を表す Gauge 群（resource × state で4系列）
var resourceState = prometheus.NewGaugeVec(
	prometheus.GaugeOpts{
		Name: "resource_state",
		Help: "0/1 representation of resource state; exactly one state should be 1 per resource (but this sample breaks it).",
	},
	[]string{"resource", "state"},
)

var states = []string{"starting", "running", "stopping", "terminated"}

func init() {
	prometheus.MustRegister(resourceState)
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// /metrics を公開
	http.Handle("/metrics", promhttp.Handler())
	go func() {
		log.Println("listening on :8080")
		log.Fatal(http.ListenAndServe(":8080", nil))
	}()

	// 1つのリソースが状態遷移するだけのループ
	resource := "A"
	// 初期状態 starting=1
	setExclusive(resource, "starting")
	time.Sleep(500 * time.Millisecond)

	// 状態遷移を繰り返す
	for {
		// ランダムに次状態へ
		next := states[rand.Intn(len(states))]
		transitionBroken(resource, next)

		time.Sleep(250 * time.Millisecond)
	}
}

// 本来の意図：resource の状態を target だけ1、他を0にする（排他）
func setExclusive(resource, target string) {
	for _, s := range states {
		v := 0.0
		if s == target {
			v = 1.0
		}
		resourceState.WithLabelValues(resource, s).Set(v)
	}
}

// 壊れる遷移：
// 1) いま 1 の状態を 0 にする
// 2) 少し待つ（この間にスクレイプされると "1 が 0個" になる）
// 3) 次状態を 1 にする
func transitionBroken(resource, next string) {
	// いま1の状態を探す（Gauge自体からは読めないので、ここでは「前回next」を追跡しない＝さらに雑）
	// 代わりに「全 state を 0 に落としてから next を 1 にする」という最悪パターンにする。
	for _, s := range states {
		resourceState.WithLabelValues(resource, s).Set(0)
	}

	// ここが地雷：この間に /metrics が読まれると "resource=A のどの状態も1ではない" が観測される
	time.Sleep(time.Duration(10+rand.Intn(50)) * time.Millisecond)

	resourceState.WithLabelValues(resource, next).Set(1)

	// さらに逆順パターンも混ぜると "1が2個" も作れる
	if rand.Intn(3) == 0 {
		// nextを1にした後、別の状態も一瞬1にしてしまう（順序が崩れる例）
		other := states[rand.Intn(len(states))]
		if other != next {
			resourceState.WithLabelValues(resource, other).Set(1)
			time.Sleep(time.Duration(5+rand.Intn(20)) * time.Millisecond)
			resourceState.WithLabelValues(resource, other).Set(0)
		}
	}

	fmt.Println("transitioned to:", next)
}
