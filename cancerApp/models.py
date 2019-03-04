from django.contrib.auth.models import AbstractUser
from django.db import models

# Create your models here.


class Doctor(AbstractUser):
    username = models.CharField(max_length=20, unique=True)
    password = models.CharField(max_length=15)
    mail = models.EmailField(max_length=30)
    phone = models.CharField(max_length=11)
    address = models.CharField(max_length=20)


class Statistics(models.Model):
    username = models.CharField(max_length=20)
    label = models.CharField(max_length=20)

