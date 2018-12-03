from django.contrib import admin
from .models import Doctor
from django.contrib import admin
from django.contrib.auth import get_user_model
from django.contrib.auth.admin import UserAdmin

from .forms import CustomUserCreationForm, CustomUserChangeForm

# Register your models here.


class CustomUserAdmin(UserAdmin):
    add_form = CustomUserCreationForm
    form = CustomUserChangeForm
    model = Doctor
    list_display = ['username', 'email', ]


admin.site.register(Doctor, CustomUserAdmin)

