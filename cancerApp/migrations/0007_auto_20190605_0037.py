# Generated by Django 2.2.1 on 2019-06-05 00:37

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('cancerApp', '0006_statistics_patient_id'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='doctor',
            name='address',
        ),
        migrations.RemoveField(
            model_name='doctor',
            name='mail',
        ),
        migrations.RemoveField(
            model_name='doctor',
            name='phone',
        ),
    ]