import os
import tempfile

import streamlit as st

from src.anpr.pipeline import recognize_vehicle_pair


st.set_page_config(
    page_title="Nepal ANPR POC",
    page_icon="🚗",
    layout="wide",
)


st.title("Nepal Vehicle Number Plate Recognition")

st.caption(
    "Proof-of-concept ANPR system for Nepali vehicle plates."
)


# ---------------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------------

left, right = st.columns(2)


with left:
    st.subheader("Front Image")

    front_file = st.file_uploader(
        "Upload front vehicle image",
        type=["jpg", "jpeg", "png"],
        key="front",
    )

    if front_file:
        st.image(
            front_file,
            use_container_width=True,
        )


with right:
    st.subheader("Rear Image")

    rear_file = st.file_uploader(
        "Upload rear vehicle image",
        type=["jpg", "jpeg", "png"],
        key="rear",
    )

    if rear_file:
        st.image(
            rear_file,
            use_container_width=True,
        )


# ---------------------------------------------------------
# RUN ANPR
# ---------------------------------------------------------

if front_file and rear_file:

    if st.button(
        "Recognize Number Plate",
        type="primary",
        use_container_width=True,
    ):

        front_suffix = os.path.splitext(
            front_file.name
        )[1]

        rear_suffix = os.path.splitext(
            rear_file.name
        )[1]

        front_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=front_suffix,
        )

        rear_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=rear_suffix,
        )

        try:
            front_temp.write(
                front_file.getvalue()
            )

            rear_temp.write(
                rear_file.getvalue()
            )

            front_temp.close()
            rear_temp.close()

            with st.spinner(
                "Detecting and recognizing plates..."
            ):
                result = recognize_vehicle_pair(
                    front_temp.name,
                    rear_temp.name,
                )

            st.divider()

            st.header("ANPR Result")

            front_result = result["front"]
            rear_result = result["rear"]

            front_column, rear_column = (
                st.columns(2)
            )

            # ---------------------------------------------
            # FRONT RESULT
            # ---------------------------------------------

            with front_column:
                st.subheader("Front Plate")

                if front_result["success"]:

                    st.success(
                        front_result[
                            "normalized_text"
                        ]
                    )

                    st.write(
                        "Detection confidence:",
                        round(
                            front_result[
                                "detection_confidence"
                            ],
                            2,
                        ),
                    )

                    st.write(
                        "OCR confidence:",
                        round(
                            front_result[
                                "ocr_confidence"
                            ],
                            2,
                        ),
                    )

                    if os.path.exists(
                        "outputs/front_plate_pipeline.png"
                    ):
                        st.image(
                            "outputs/front_plate_pipeline.png",
                            caption="Detected front plate",
                            use_container_width=True,
                        )

                else:
                    st.warning(
                        front_result["error"]
                    )

                    if (
                        "detection_confidence"
                        in front_result
                    ):
                        st.write(
                            "Detection confidence:",
                            round(
                                front_result[
                                    "detection_confidence"
                                ],
                                2,
                            ),
                        )

            # ---------------------------------------------
            # REAR RESULT
            # ---------------------------------------------

            with rear_column:
                st.subheader("Rear Plate")

                if rear_result["success"]:

                    st.success(
                        rear_result[
                            "normalized_text"
                        ]
                    )

                    st.write(
                        "Detection confidence:",
                        round(
                            rear_result[
                                "detection_confidence"
                            ],
                            2,
                        ),
                    )

                    st.write(
                        "OCR confidence:",
                        round(
                            rear_result[
                                "ocr_confidence"
                            ],
                            2,
                        ),
                    )

                    if os.path.exists(
                        "outputs/rear_plate_pipeline.png"
                    ):
                        st.image(
                            "outputs/rear_plate_pipeline.png",
                            caption="Detected rear plate",
                            use_container_width=True,
                        )

                else:
                    st.warning(
                        rear_result["error"]
                    )

            # ---------------------------------------------
            # PAIR COMPARISON
            # ---------------------------------------------

            st.divider()

            st.subheader(
                "Front / Rear Comparison"
            )

            match = result["match"]

            if match is None:
                st.warning(
                    "Not enough OCR information to compare both plates."
                )

            else:
                similarity = match[
                    "similarity"
                ]

                st.metric(
                    "Similarity",
                    f"{similarity * 100:.0f}%",
                )

                if (
                    match["status"]
                    == "exact_match"
                ):
                    st.success(
                        "Exact front/rear match"
                    )

                elif (
                    match["status"]
                    == "possible_match"
                ):
                    st.warning(
                        "Possible match — manual verification recommended"
                    )

                else:
                    st.error(
                        "Front and rear OCR results do not match"
                    )

            # ---------------------------------------------
            # POC NOTE
            # ---------------------------------------------

            st.info(
                "POC limitation: plate detection is currently "
                "more reliable than OCR. Predictions should be "
                "manually verified, especially for low-resolution, "
                "glare, or unusual plate layouts."
            )

        finally:
            if os.path.exists(
                front_temp.name
            ):
                os.unlink(
                    front_temp.name
                )

            if os.path.exists(
                rear_temp.name
            ):
                os.unlink(
                    rear_temp.name
                )