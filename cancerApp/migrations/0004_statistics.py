# Generated by Django 2.1.5 on 2019-03-04 13:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cancerApp', '0003_auto_20181123_1507'),
    ]

    operations = [
        migrations.CreateModel(
            name='Statistics',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(max_length=20, unique=True)),
                ('label', models.CharField(max_length=20)),
            ],
        ),
    ]