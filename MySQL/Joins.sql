SELECT s.student_id, s.name, m.subject, m.marks
FROM students s
LEFT JOIN marks m
ON s.student_id = m.student_id;
