from django.contrib.auth.models import AbstractUser
from django.db import models

# Create your models here.


class Doctor(AbstractUser):
    username = models.CharField(max_length=20, unique=True)
    password = models.TextField()


class Statistics(models.Model):
    patient_id = models.CharField(max_length=20, default='0')
    username = models.CharField(max_length=20)
    label = models.CharField(max_length=20)

    def get_stats(self,username=any,patient_id=any,label=any):
       return self.objects.filter(username=username,patient_id=patient_id,label=label)


