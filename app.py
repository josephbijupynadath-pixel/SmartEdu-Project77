import io
import pandas as pd
import streamlit as st

from modules.recommendation import recommend_resources
from modules.prediction import predict_marks, load_training_data, evaluate_model
from modules.blockchain import create_block, verify_blockchain

st.set_page_config(page_title="SmartEdu", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.main { background-color: #f5f7fb; }
.block-container { padding-top: 2rem; }
h1 { color: #183153; }
</style>
""", unsafe_allow_html=True)

REQUIRED_COLUMNS = ["Department", "Semester", "Subject", "Marks", "Attendance"]

def read_uploaded_dataset(uploaded_file):
    """Read a user-uploaded CSV or Excel dataset."""
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Upload a CSV or Excel file.")

st.sidebar.title("🎓 SmartEdu")
st.sidebar.caption("Smart Education System")
st.sidebar.divider()

st.sidebar.subheader("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Choose a student dataset",
    type=["csv", "xlsx", "xls"],
    help="Upload a CSV or Excel file containing student academic records."
)

if uploaded_file is not None:
    try:
        uploaded_df = read_uploaded_dataset(uploaded_file)
        missing = [col for col in REQUIRED_COLUMNS if col not in uploaded_df.columns]
        if missing:
            st.sidebar.error("Missing required columns: " + ", ".join(missing))
            st.session_state.pop("student_dataset", None)
            st.session_state.pop("dataset_name", None)
        elif uploaded_df.empty:
            st.sidebar.error("The uploaded dataset is empty.")
            st.session_state.pop("student_dataset", None)
            st.session_state.pop("dataset_name", None)
        else:
            st.session_state["student_dataset"] = uploaded_df
            st.session_state["dataset_name"] = uploaded_file.name
            st.sidebar.success(f"Loaded {len(uploaded_df)} student records.")
    except Exception as exc:
        st.sidebar.error(f"Could not read file: {exc}")
        st.session_state.pop("student_dataset", None)
        st.session_state.pop("dataset_name", None)

df = st.session_state.get("student_dataset")
dataset_name = st.session_state.get("dataset_name")

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "Home",
        "Dataset Upload",
        "Student Dashboard",
        "AI Prediction",
        "Learning Resources",
        "Blockchain Security",
        "Research Results"
    ]
)
st.sidebar.divider()
st.sidebar.info("AI-Based Personalized Learning & Secure Academic Records")

def require_dataset():
    if df is None:
        st.warning("Please upload a student dataset using the uploader in the sidebar.")
        st.info("Supported formats: CSV (.csv) and Excel (.xlsx, .xls).")
        st.markdown(
            "**Required columns:** `Department`, `Semester`, `Subject`, `Marks`, `Attendance`."
        )
        return False
    return True

if page == "Home":
    st.title("🎓 Welcome to SmartEdu")
    st.subheader("AI-Based Personalized Learning and Blockchain-Secured Academic Records")
    st.write(
        "SmartEdu is an intelligent educational platform designed to support personalized "
        "learning, student performance prediction and secure academic record verification. "
        "Upload your own student dataset to begin."
    )
    st.divider()
    st.subheader("Application Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📊 Student Dashboard")
        st.write("Monitor student learning and academic performance.")
    with col2:
        st.info("🤖 AI Prediction")
        st.write("Predict student performance using Machine Learning.")
    with col3:
        st.info("📚 Personalized Learning")
        st.write("Provide suitable learning resources.")
    col4, col5 = st.columns(2)
    with col4:
        st.info("🔐 Blockchain Security")
        st.write("Verify the integrity of academic records.")
    with col5:
        st.info("📈 Research Results")
        st.write("Evaluate model performance and research findings.")
    st.divider()
    if df is None:
        st.warning("No dataset uploaded yet. Use the uploader in the sidebar.")
    else:
        st.success(f"Dataset ready: {dataset_name} ({len(df)} records).")

elif page == "Dataset Upload":
    st.title("📂 Dataset Upload")
    st.write("Upload your own student academic dataset. The uploaded file is used throughout SmartEdu.")
    if require_dataset():
        st.success(f"Currently using: {dataset_name}")
        c1, c2 = st.columns(2)
        c1.metric("Total Records", len(df))
        c2.metric("Total Columns", len(df.columns))
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)
        st.subheader("Column Information")
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [str(dtype) for dtype in df.dtypes],
            "Missing Values": [int(df[col].isna().sum()) for col in df.columns]
        })
        st.dataframe(info_df, use_container_width=True)
    else:
        st.caption("Once a valid file is uploaded in the sidebar, its preview and details will appear here.")
    with st.expander("Dataset format requirements"):
        st.write("Required columns for SmartEdu's dashboard and AI modules:")
        st.code("Department, Semester, Subject, Marks, Attendance")
        st.write("An optional `Student_Name` column is used for chart labels. CSV and Excel files are supported.")

elif page == "Student Dashboard":
    st.title("📊 Student Dashboard")
    st.write("Student academic performance overview.")
    if require_dataset():
        clean_df = df.copy()
        for col in ["Marks", "Attendance"]:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
        clean_df = clean_df.dropna(subset=["Marks", "Attendance", "Department"])
        if clean_df.empty:
            st.error("No usable records found. Check the Marks, Attendance and Department columns.")
        else:
            st.subheader("Student Statistics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Students", len(df))
            col2.metric("Average Marks", round(clean_df["Marks"].mean(), 2))
            col3.metric("Average Attendance", round(clean_df["Attendance"].mean(), 2))
            col4.metric("Departments", clean_df["Department"].nunique())
            st.subheader("Student Academic Records")
            st.dataframe(df, use_container_width=True)
            label_col = "Student_Name" if "Student_Name" in clean_df.columns else None
            if label_col:
                chart_df = clean_df.set_index(label_col)
                st.subheader("Student Marks Analysis")
                st.bar_chart(chart_df["Marks"])
                st.subheader("Student Attendance Analysis")
                st.bar_chart(chart_df["Attendance"])
            else:
                st.info("Add an optional Student_Name column to display student names on charts.")
                st.subheader("Student Marks Analysis")
                st.bar_chart(clean_df["Marks"])
                st.subheader("Student Attendance Analysis")
                st.bar_chart(clean_df["Attendance"])

elif page == "AI Prediction":
    st.title("🤖 AI-Based Student Performance Prediction")
    st.write("Predict student academic performance using the uploaded dataset.")
    if require_dataset():
        try:
            training_df = load_training_data(df)
            st.subheader("Enter Student Details")
            col1, col2 = st.columns(2)
            with col1:
                department = st.selectbox("Select Department", sorted(training_df["Department"].astype(str).unique()))
                semester = st.selectbox("Select Semester", sorted(training_df["Semester"].unique()))
            with col2:
                subject = st.selectbox("Select Subject", sorted(training_df["Subject"].astype(str).unique()))
                attendance = st.slider("Attendance Percentage", 0, 100, 80)
            if st.button("Predict Student Performance"):
                marks = predict_marks(department, semester, subject, attendance, training_df)
                st.subheader("Prediction Results")
                st.metric("Predicted Marks", f"{marks}/100")
                if marks >= 85:
                    category = "Excellent"
                    st.success("Excellent Performance!")
                elif marks >= 70:
                    category = "Good"
                    st.success("Good Performance!")
                elif marks >= 50:
                    category = "Average"
                    st.warning("Average Performance")
                else:
                    category = "At Risk"
                    st.error("Student Needs Academic Support")
                st.write("Performance Category:", category)
                st.progress(int(marks))
                st.info("Recommendation: Continue regular learning and maintain good attendance.")
            st.caption(f"Model trained using {len(training_df)} valid student records from {dataset_name}.")
        except Exception as exc:
            st.error(f"Prediction error: {exc}")

elif page == "Learning Resources":
    st.title("📚 Personalized Learning Resources")
    st.write("Learning materials based on student academic needs.")
    subject_options = sorted(df["Subject"].dropna().astype(str).unique()) if df is not None else ["Database", "Python", "Mathematics", "Computer Networks"]
    subject = st.selectbox("Select Subject", subject_options)
    marks = st.slider("Enter Student Marks", 0, 100, 60)
    attendance = st.slider("Attendance Percentage", 0, 100, 80)
    if st.button("Get Learning Recommendations"):
        recommendations = recommend_resources(subject, marks, attendance)
        st.subheader("Recommended Learning Resources")
        for item in recommendations:
            st.write("📌", item)

elif page == "Blockchain Security":
    st.title("🔒 Blockchain Academic Record Security")
    st.write(
        "Create a tamper-evident blockchain record directly from a student "
        "in your uploaded dataset. The grade is calculated automatically from marks."
    )

    if require_dataset():
        blockchain_df = df.copy()
        blockchain_df["Marks"] = pd.to_numeric(blockchain_df["Marks"], errors="coerce")
        blockchain_df["Attendance"] = pd.to_numeric(
            blockchain_df["Attendance"], errors="coerce"
        )
        blockchain_df = blockchain_df.dropna(
            subset=["Marks", "Attendance", "Department", "Semester", "Subject"]
        ).copy()

        if blockchain_df.empty:
            st.error(
                "The uploaded dataset has no usable records. Check that Marks, "
                "Attendance, Department, Semester and Subject contain valid values."
            )
        else:
            def calculate_grade(mark):
                if mark >= 90:
                    return "A+"
                if mark >= 80:
                    return "A"
                if mark >= 70:
                    return "B"
                if mark >= 60:
                    return "C"
                if mark >= 50:
                    return "D"
                if mark >= 40:
                    return "E"
                return "F"

            st.subheader("Create Secure Academic Record")
            st.caption(
                "Choose a student from the uploaded dataset. Student details, marks "
                "and attendance will be populated automatically."
            )
            name_column = "Student_Name" if "Student_Name" in blockchain_df.columns else None
            id_column = next(
                (col for col in ["Student_ID", "StudentId", "ID", "student_id"]
                 if col in blockchain_df.columns),
                None
            )
            row_indices = blockchain_df.index.tolist()

            def student_option(row_index):
                row = blockchain_df.loc[row_index]
                name = (
                    str(row[name_column])
                    if name_column and pd.notna(row[name_column])
                    else f"Student record {row_index + 1}"
                )
                identifier = (
                    str(row[id_column])
                    if id_column and pd.notna(row[id_column])
                    else f"STU-{row_index + 1:04d}"
                )
                return f"{name} | {identifier} | {row['Subject']}"

            selected_index = st.selectbox(
                "Select Student", row_indices,
                format_func=student_option,
                key="blockchain_student_selection"
            )
            selected = blockchain_df.loc[selected_index]
            student_name = (
                str(selected[name_column])
                if name_column and pd.notna(selected[name_column])
                else f"Student record {selected_index + 1}"
            )
            student_id = (
                str(selected[id_column])
                if id_column and pd.notna(selected[id_column])
                else f"STU-{selected_index + 1:04d}"
            )
            original_marks = float(selected["Marks"])
            attendance = float(selected["Attendance"])

            st.subheader("Edit Academic Marks")
            st.caption(
                "Marks are editable here. The grade updates automatically. "
                "Saving creates a new blockchain entry; previous entries are preserved."
            )
            marks = st.number_input(
                "Marks (out of 100)",
                min_value=0.0,
                max_value=100.0,
                value=max(0.0, min(100.0, original_marks)),
                step=1.0,
                key=f"blockchain_marks_{selected_index}",
            )
            grade = calculate_grade(marks)
            reason = st.text_input(
                "Correction reason (optional)",
                placeholder="e.g. Marks corrected after re-evaluation",
                key=f"blockchain_reason_{selected_index}",
            )

            preview_col1, preview_col2, preview_col3 = st.columns(3)
            preview_col1.metric("Student", student_name)
            preview_col2.metric(
                "Marks", f"{marks:g}/100",
                delta=f"{marks - original_marks:+g} from dataset" if marks != original_marks else None,
            )
            preview_col3.metric("Calculated Grade", grade)
            st.write({
                "Student ID": student_id,
                "Department": str(selected["Department"]),
                "Semester": str(selected["Semester"]),
                "Subject": str(selected["Subject"]),
                "Attendance": f"{attendance:g}%",
                "Original dataset marks": f"{original_marks:g}/100",
            })

            if st.button("Save Marks to Blockchain", type="primary"):
                record = {
                    "student_id": student_id,
                    "name": student_name,
                    "department": str(selected["Department"]),
                    "semester": str(selected["Semester"]),
                    "subject": str(selected["Subject"]),
                    "marks": float(marks),
                    "attendance": attendance,
                    "grade": grade,
                    "correction_reason": reason.strip() or "Initial record / marks update",
                }
                chain = st.session_state.setdefault("blockchain", [])
                previous_hash = chain[-1]["hash"] if chain else "0"
                new_block = create_block(record, previous_hash)
                chain.append(new_block)
                st.success(
                    "Academic record added to the blockchain. "
                    "Earlier records have been preserved."
                )
                st.write("Generated SHA-256 Hash:")
                st.code(new_block["hash"])
                st.json(new_block["record"])

            st.caption(
                "Grade scale: A+ (90–100), A (80–89), B (70–79), "
                "C (60–69), D (50–59), E (40–49), F (below 40)."
            )

    st.divider()
    st.subheader("Verify Academic Record")
    if "blockchain" in st.session_state:
        if st.button("Verify Blockchain"):
            if verify_blockchain(st.session_state["blockchain"]):
                st.success("Blockchain is valid. Record integrity verified!")
            else:
                st.error("Blockchain verification failed!")
        with st.expander("View stored blockchain"):
            st.json(st.session_state["blockchain"])
    else:
        st.info("Upload a dataset and generate an academic record first.")

elif page == "Research Results":
    st.title("📊 Research Results")
    st.write("Machine Learning Model Evaluation and Research Analysis.")
    if require_dataset():
        try:
            results = evaluate_model(df)
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", results["MAE"])
            m2.metric("RMSE", results["RMSE"])
            m3.metric("R² Score", results["R2 Score"])
            c1, c2 = st.columns(2)
            c1.metric("Training Records", results["Training Records"])
            c2.metric("Testing Records", results["Testing Records"])
            st.subheader("Actual vs. Predicted Marks")
            comparison_df = results["Comparison"]
            st.dataframe(comparison_df, use_container_width=True)
            st.bar_chart(comparison_df)
        except Exception as exc:
            st.error(f"Error displaying research results: {exc}")
