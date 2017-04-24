package com.taot.demo.web;

import org.springframework.web.servlet.ModelAndView;
import org.springframework.web.servlet.handler.HandlerInterceptorAdapter;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

@SuppressWarnings("unused")
public class LoggingInterceptor extends HandlerInterceptorAdapter {

    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
        System.out.println("REQUEST: " + request.getMethod() + " " + request.getServletPath());
        request.setAttribute("start_time", System.currentTimeMillis());
        return super.preHandle(request, response, handler);
    }

    @Override
    public void postHandle(HttpServletRequest request, HttpServletResponse response, Object handler, ModelAndView modelAndView) throws Exception {
        // TODO postHandle not working, why
        System.out.println("postHandler");
        super.postHandle(request, response, handler, modelAndView);
    }

    @Override
    public void afterCompletion(HttpServletRequest request, HttpServletResponse response, Object handler, Exception ex) throws Exception {
        long startTime = (Long) request.getAttribute("start_time");
        long duration = System.currentTimeMillis() - startTime;
        System.out.println("RESPONSE: " + request.getMethod() + " " + request.getServletPath() + ": " + duration + " ms");
        super.afterCompletion(request, response, handler, ex);
    }
}
