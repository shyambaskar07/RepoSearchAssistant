from django.db import models
class UploadedFile(models.Model):
    file = models.FileField(upload_to="uploaded_files/")
    name = models.CharField(max_length=400, blank=True)
    text = models.TextField(blank=True)
    vector = models.BinaryField(blank=True, null=True)  # store embedding bytes
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        if not self.name:
            self.name = self.file.name
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name
