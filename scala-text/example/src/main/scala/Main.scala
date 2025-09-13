def swapArray[T] (arr: Array[T])(i: Int, j: Int): Unit =
  val tmp = arr(i)
  arr(i) = arr(j)
  arr(j) = tmp
@main def run(): Unit =
  val arr = Array(1,2,3,4,5)
  swapArray(arr)(0,1)
  println(arr.foreach(println))
