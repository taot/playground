import Dependencies._

lazy val root = (project in file(".")).settings(
  inThisBuild(List(
    organization := "com.github.taot",
    scalaVersion := "2.12.1",
    version      := "0.1.0-SNAPSHOT"
  )),
  name := "http-api-doc",
  libraryDependencies ++= Seq(
    "org.yaml" % "snakeyaml" % "1.17",
    scalaTest % Test
  )
)
