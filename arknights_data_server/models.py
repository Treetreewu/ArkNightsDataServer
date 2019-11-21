from django.contrib.auth.base_user import AbstractBaseUser
from django.db import models


class Employee(models.Model):
    level = models.IntegerField()


class User(AbstractBaseUser):
    wechat_uid = None









