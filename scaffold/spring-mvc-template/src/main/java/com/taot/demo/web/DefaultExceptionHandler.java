package com.taot.demo.web;

import com.taot.demo.model.ResponseWrapper;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.ResponseStatus;

import javax.servlet.http.HttpServletRequest;

@ControllerAdvice
public class DefaultExceptionHandler {

    @ResponseStatus(value = HttpStatus.INTERNAL_SERVER_ERROR)
    @ExceptionHandler(value = Exception.class)
    @ResponseBody
    public ResponseWrapper defaultExceptionHandler(HttpServletRequest req, Exception e) {
        return new ResponseWrapper("ERROR", e.getMessage());
    }
}
