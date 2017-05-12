package com.taot.demo.web;

import com.taot.demo.model.User;
import com.taot.demo.validator.UserValidator;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.*;

import javax.annotation.PostConstruct;
import javax.servlet.http.HttpServletRequest;

@Controller
@RequestMapping(value = "/test")
@SuppressWarnings("unused")
public class TestController {

    @Autowired
    private UserValidator userValidator;

    @PostConstruct
    public void init() {
        System.out.println("++++++++++++++++++++ TestController.init ++++++++++++++");
    }

    @RequestMapping(method = RequestMethod.GET, value = "")
    @ResponseBody
    public User testGet(
            @RequestParam(value = "id", required = false, defaultValue = "1") int id,
            @RequestParam(value = "name") String name,
            HttpServletRequest request) {
        return new User(id, name);
    }

    @RequestMapping(method = RequestMethod.POST, value = "")
    @ResponseBody
    public User testPost(@ModelAttribute("user") User user, BindingResult result) {
        userValidator.validate(user, result);
        System.out.println(user);
        System.out.println(result);
        return user;
    }

    @RequestMapping(method = RequestMethod.GET, value = "/error")
    @ResponseBody
    public User testError(HttpServletRequest request) {
        throw new RuntimeException("test error");
    }
}
