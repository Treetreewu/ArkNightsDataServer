from django.contrib.auth.base_user import AbstractBaseUser
from django.db import models

from data_bus.wiki_spider import DoMan


class Employee(models.Model):

    level = models.IntegerField()


class User(AbstractBaseUser):
    wechat_uid = None













