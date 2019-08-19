#!/usr/bin/env bash


# video! 用在VBT上效果很差，不知道为什么

--img_size 361 --cls_list wild_before,RE_ORG --batch_size 16 --test_model 66_2520

--img_size 361 --cls_list A_before,RE_ORG --batch_size 1 --test_model 66_2520


--img_size 256 --cls_list wild_256,RE_REF --batch_size 1 --test_model 66_2520


# 测试一下在有妆图片下的效果

--img_size 256 --cls_list RE_REF,RE_ORI --batch_size 1 --test_model 66_2520

--img_size 256 --cls_list RE_ORG,wild_256 --batch_size 1 --test_model 66_2520


# new

--task_name default --cls_list wild_256,RE_REF --batch_size 1 --test_model 26_2520