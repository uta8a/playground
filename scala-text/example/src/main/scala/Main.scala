trait Logger(prefix: String):
  // 本体が「トレイトの初期化コード」= コンストラクタに相当
  def log(msg: String): Unit =
    println(s"$prefix$msg")

class Service extends Logger("SVC: "):
  def work(): Unit = log("doing work")

@main def demo(): Unit =
  new Service().work()
// => SVC: doing work

