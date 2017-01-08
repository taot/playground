package com.taot.apidoc

import java.io.{File, FileInputStream}
import org.yaml.snakeyaml.Yaml

object Hello {
  def main(args: Array[String]) {
    println("Hello");
    val inputStream = new FileInputStream(new File("D:\\gitoschina\\salum-baby-api\\SalumBabyAPI.yaml"));
    val yaml = new Yaml();
    val obj = yaml.load(inputStream);
    println(obj.getClass());
  }
}
