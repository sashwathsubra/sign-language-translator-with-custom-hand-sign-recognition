from sign_language_translator.demo.pipeline import DemoPipeline


def main() -> None:
    pipeline = DemoPipeline()

    sample_landmarks = [
        [0.0, 0.0, 0.0],
        [0.05, 0.05, 0.0],
        [0.10, 0.10, 0.0],
        [0.14, 0.14, 0.0],
        [0.18, 0.18, 0.0],
        [0.22, 0.22, 0.0],
    ]

    token, confidence = pipeline.classifier.predict([sample_landmarks])
    sentence = pipeline.process_gloss(token)
    print(f"token={token} confidence={confidence:.2f}")
    print(f"sentence={sentence}")


if __name__ == "__main__":
    main()
