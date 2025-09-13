trait A:
  def who = "A"

trait B:
  def who = "B"

trait C:
  def who = "C"

class X extends A, B, C: // 左→右の順で線形化（後勝ち）
  override def who =
    // 必要なら特定のトレイト経由で super 呼び分け
    val fromB = super[B].who
    val fromC = super[C].who
    s"X(${fromB}|${fromC})"  // => "X(B|C)"

@main def run(): Unit =
  println(new X().who)

