trait Greeter:
  def greet(name: String): String

val g = new Greeter:
  def greet(name: String) = s"Hello, $name"

@main def run(): Unit =
  println(g.greet("world"))

