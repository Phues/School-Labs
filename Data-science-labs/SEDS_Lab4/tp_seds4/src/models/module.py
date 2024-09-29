def serve_beer(age):
  if (age is None) or (age<18):
    return "No beer"
  else:
    return "Have beer"