val scala3Version = "3.7.2"

lazy val root = project
  .in(file("."))
  .settings(
    name := "example",
    version := "0.1.0-SNAPSHOT",

    scalaVersion := scala3Version,

    libraryDependencies ++= Seq(
      "org.scalameta" %% "munit" % "1.0.0" % Test,
      "org.scalatest" %% "scalatest-flatspec" % "3.2.19" % "test",
      "org.scalatest" %% "scalatest-diagrams" % "3.2.19" % "test",
    )
  )
scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked")
