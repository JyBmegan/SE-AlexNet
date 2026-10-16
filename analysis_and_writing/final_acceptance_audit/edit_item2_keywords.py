from pathlib import Path

from docx import Document


DOCX_PATH = Path(
    "/Users/Megan/0_MyFolders/0_Projects/投稿SEAlexNet/最后修改/xb26-0201_正文.docx"
)
OLD_TEXT = "关键词 表征压缩，卷积神经网络，视知觉，面孔认知，认知机制"
NEW_TEXT = "关键词：表征压缩，卷积神经网络；视知觉；面孔认知；认知机制"


def main() -> None:
    document = Document(DOCX_PATH)
    matches = [paragraph for paragraph in document.paragraphs if paragraph.text == OLD_TEXT]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one keyword paragraph, found {len(matches)}")

    paragraph = matches[0]
    if len(paragraph.runs) != 1:
        raise RuntimeError(f"Expected one run in keyword paragraph, found {len(paragraph.runs)}")

    paragraph.runs[0].text = NEW_TEXT
    document.save(DOCX_PATH)


if __name__ == "__main__":
    main()
