from pathlib import Path
import cv2
import os

class Label:
    def __init__(self, path: str):
        self.path = path

    def annotate_images(self,input_img_paths: list[Path], output_path: Path, labels: list[str]) -> None:

        for idx, img_path in enumerate(input_img_paths):
            img = cv2.imread(str(img_path))

            cv2.putText(
                img,
                f"id: {img_path}, liko: {len(input_img_paths) - idx},",
                (00, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            #add label help text
            for i, label in enumerate(labels, start=1):
                cv2.putText(
                    img,
                    f"{i}: {label},",
                    (10, 30 + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )


            labels_key_dict = {ord(str(i)): label for i, label in enumerate(labels, start=1)}

            #create all classification folders
            for label in labels:
                label_dir = output_path / label
                label_dir.mkdir(parents=True, exist_ok=True)

            cv2.imshow("Image", img)

            while True:
                key = cv2.waitKey(0)

                # Quit Annotation Tool
                if key == ord("q"):
                    return

                if key in labels_key_dict:
                    label = labels_key_dict[key]
                    print(f"Classified as {label}")

                    output_img_path = output_path / label / img_path.name
                    img_path.rename(output_img_path)
                    break

            cv2.destroyAllWindows()

    def run(self):
        input_path = Path(os.path.join(self.path, "raw"))
        input_img_paths = sorted(input_path.glob("*.jpg"))

        output_path = Path(self.path)
        output_path.mkdir(parents=True, exist_ok=True)

        self.annotate_images(
            input_img_paths=input_img_paths,
            output_path=output_path,
            labels=["good", "anomaly"],
        )