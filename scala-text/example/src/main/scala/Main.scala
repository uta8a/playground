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
  println("Hello world!")
  println(msg)

def msg = "I was compiled by Scala 3. :)"
