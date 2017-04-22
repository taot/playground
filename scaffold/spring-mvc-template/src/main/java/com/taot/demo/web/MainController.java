package com.taot.demo.web;

import com.taot.demo.model.User;
import com.taot.demo.validator.UserValidator;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpServletRequest;

@Controller
@RequestMapping(value = "/")
public class MainController {

    @Autowired
    private UserValidator userValidator;

    public MainController() {
        System.out.println("Main controller constructor");
    }

    @RequestMapping(method = RequestMethod.GET, value = "/test")
    @ResponseBody
    public User testGet(
            @RequestParam(value = "id", required = false, defaultValue = "1") int id,
            @RequestParam(value = "name") String name,
            HttpServletRequest request) {

        return new User(id, name);
    }

    @RequestMapping(method = RequestMethod.POST, value = "/test")
    @ResponseBody
    public User testPost(@ModelAttribute("user") User user, BindingResult result) {
//        return new User(1, "Terry");
        userValidator.validate(user, result);
        System.out.println(user);
        System.out.println(result);
        return user;
    }
}
