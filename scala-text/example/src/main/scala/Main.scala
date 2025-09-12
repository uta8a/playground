def firstLastSame[T](l: List[T]): Boolean = {
  l.head == l.last
}

def printfirstLastSame1000Times(): Unit = {
  for (i <- 1 to 1000) {
    val head = new scala.util.Random(new java.security.SecureRandom()).alphanumeric.take(5).toList
    head match {
      case List(a,b,c,d,e) => println(s"$a$b$c$d$a")
    }
  }
}
var age: Int = 5
var isSchoolStarted: Boolean = false

def isChild(age: Int, isSchoolStarted: Boolean): Unit = {
  if ((1 <= age && age <= 6) && !isSchoolStarted) {
    println("幼児です")
  } else {
    println("幼児ではありません")
  }
}

class User(val name: String, val age: Int)

object User {
  def printUser(user: User) = println(user.name + " " + user.age)
}

@main def hello(): Unit =
  isChild(age, isSchoolStarted)
  printfirstLastSame1000Times()
