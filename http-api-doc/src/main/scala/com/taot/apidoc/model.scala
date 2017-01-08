package com.taot.apidoc

package object model {

  case class ApiDoc(
    title: String,
    version: String,
    basePath: String,
    description: String,
    paths: Seq[Iface],
    definitions: Seq[Def]
  )

  case class Iface(
    path: String,
    method: String,
    description: String,
    request: Request,
    responses: Seq[Response]
  )

  case class Request(
    description: String,
    query: Seq[Param],
    headers: Seq[Param],
    body: Option[Body],
    example: String
  )

  case class Response(
    statusCode: String,
    description: String,
    headers: Seq[Param],
    body: Option[Body],
    example: String
  )

  case class Param(
    name: String,
    `type`: String,
    format: String,
    description: String
  )

  case class Body(
    `type`: String,
    form: Seq[Param],
    raw: String
  )

  case class Def(

  )
}
