from preprocessing import normalize_image
from inference import extract_model
from postprocessing import predict

#input path
path = '/9j/4AAQSkZJRgABAQAAAQABAAD/4gIoSUNDX1BST0ZJTEUAAQEAAAIYAAAAAAQwAABtbnRyUkdCIFhZWiAAAAAAAAAAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAAHRyWFlaAAABZAAAABRnWFlaAAABeAAAABRiWFlaAAABjAAAABRyVFJDAAABoAAAAChnVFJDAAABoAAAAChiVFJDAAABoAAAACh3dHB0AAAByAAAABRjcHJ0AAAB3AAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAFgAAAAcAHMAUgBHAEIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFhZWiAAAAAAAABvogAAOPUAAAOQWFlaIAAAAAAAAGKZAAC3hQAAGNpYWVogAAAAAAAAJKAAAA+EAAC2z3BhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABYWVogAAAAAAAA9tYAAQAAAADTLW1sdWMAAAAAAAAAAQAAAAxlblVTAAAAIAAAABwARwBvAG8AZwBsAGUAIABJAG4AYwAuACAAMgAwADEANv/bAIQAAwICCAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgHCggICAgJCQkHBwsNCggNBwgJCAEDBAQGBQYIBQUICAcHBwgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI/8AAEQgDPAM8AwEiAAIRAQMRAf/EAB4AAQABBAMBAQAAAAAAAAAAAAADAQIEBQYHCQgK/8QARRAAAgEDAgQDBQUGBQIGAQUAAAECAwQRITEFEkFRBgdhCBMicYEJFDKRoUJSscHR8BUjYuHxgpIWJDNDcsJjNFNzhNL/xAAaAQEBAQEBAQEAAAAAAAAAAAAAAQIDBAUG/8QAKBEBAQEAAQQCAQQCAwEAAAAAAAERAgMSITEEQVEiYXGBEzJCwdEU/9oADAMBAAIRAxEAPwD1TAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA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AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH//2Q=='
# path = "jury test set/sandal_1.jpeg"

if __name__ == "__main__":
    img = normalize_image(path)
    model = extract_model()
    predict(model, img)