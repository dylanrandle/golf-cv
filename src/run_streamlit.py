import streamlit as st
import pandas as pd

from process_poses import (
    get_keypoints,
    load_poses,
    KEYPOINT_TO_DESCRIPTION_MAP,
)


def render_top_text():
    st.title("Golf Swing Analysis with CV")
    st.subheader("Preamble")
    st.markdown(
        "The goal of this project is to leverage 3D human pose estimation, a computer vision technique "
        "which predicts a set of keypoints on a human body in an image or video, to gather insights about "
        "my golf swing and hopefully help me improve it."
    )


def render_video_selection():
    selection = st.selectbox(
        label="Choose a video", options=["four_iron_1", "four_iron_2"]
    )
    st.video(f"data/videos/{selection}.MOV")
    return selection


def render_keypoint_line_plots(video_id: str):
    st.subheader("Keypoints")
    st.markdown(
        "Below we can see the predicted keypoint positions for each frame in the video."
    )

    poses = load_poses(f"data/predictions/{video_id}.json")
    keypoints = get_keypoints(poses)

    for axis_idx, axis_name in enumerate(["x", "y", "z"]):
        axis_keypoints = pd.DataFrame(
            keypoints[:, :, axis_idx], columns=KEYPOINT_TO_DESCRIPTION_MAP.values()
        )
        # TODO: fix legend off edges
        st.line_chart(
            axis_keypoints,
            x_label="Frame Index",
            y_label=f"Position ({axis_name})",
        )


def main():
    st.set_page_config(page_title="Golf CV")
    render_top_text()
    selected_video_id = render_video_selection()
    render_keypoint_line_plots(selected_video_id)
    st.info("This is a work in progress. Check back soon!")


if __name__ == "__main__":
    main()
