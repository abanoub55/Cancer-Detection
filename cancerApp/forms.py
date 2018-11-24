from django import forms
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from .models import Doctor


class CustomUserCreationForm(UserCreationForm):

    class Meta(UserCreationForm):
        model = Doctor
        fields = ('username',)


class CustomUserChangeForm(UserChangeForm):

    class Meta:
        model = Doctor
        fields = ('username',)
