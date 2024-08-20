from django.db import models
from django.utils import timezone

class Film(models.Model):
    title = models.CharField(max_length=100, default = "Unknown Title")
    genre = models.CharField(max_length=100, default = "Unknown Genre")
    release_date = models.DateField(default = timezone.now)
    director = models.CharField(max_length=100, default = "Unknown Director")

    def __str__(self):
        return self.title