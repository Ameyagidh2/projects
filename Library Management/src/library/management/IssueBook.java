/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package library.management;

/**
 *
 * @author ameya
 */

import java.awt.*;
import javax.swing.*;
import java.sql.*;
import java.awt.event.*;
import javax.swing.border.*;
import java.util.*;
import com.toedter.calendar.JDateChooser;
import library.management.Home;
import library.management.conn;
public class IssueBook extends JFrame implements ActionListener{
    private JPanel contentPane;
    private JTextField t1,t2,t3,t4,t5,t6,t7,t_1,t_2,t_3,t_4,t_5,t_6,t_7;
    private JButton b1,b_1,b_2,b_4;
    JDateChooser dateChooser;
        
    public static void main(String[] args) {
	new IssueBook().setVisible(true);
    }

    public IssueBook() {
        setBounds(600, 200, 918, 442);
	contentPane = new JPanel();
	contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
	setContentPane(contentPane);
	contentPane.setLayout(null);

	JLabel l1 = new JLabel("Student_id");
	l1.setForeground(new Color(47, 79, 79));
	l1.setFont(new Font("Tahoma", Font.BOLD, 14));
	l1.setBounds(73, 50, 90, 22);
	contentPane.add(l1);

	JLabel l2 = new JLabel("Name");
	l2.setForeground(new Color(47, 79, 79));
	l2.setFont(new Font("Tahoma", Font.BOLD, 14));
	l2.setBounds(73, 80, 90, 22);
	contentPane.add(l2);

	JLabel l3 = new JLabel("Father's Name");
	l3.setForeground(new Color(47, 79, 79));
	l3.setFont(new Font("Tahoma", Font.BOLD, 12));
	l3.setBounds(73, 120, 90, 22);
	contentPane.add(l3);

	JLabel l4 = new JLabel("Course");
	l4.setForeground(new Color(47, 79, 79));
	l4.setFont(new Font("Tahoma", Font.BOLD, 14));
	l4.setBounds(73, 152, 90, 22);
	contentPane.add(l4);

	JLabel l5 = new JLabel("Branch");
	l5.setForeground(new Color(47, 79, 79));
	l5.setFont(new Font("Tahoma", Font.BOLD, 14));
	l5.setBounds(73, 181, 90, 22);
	contentPane.add(l5);

	JLabel l6 = new JLabel("Year");
	l6.setForeground(new Color(47, 79, 79));
	l6.setFont(new Font("Tahoma", Font.BOLD, 14));
	l6.setBounds(73, 205, 90, 22);
	contentPane.add(l6);

	JLabel l7 = new JLabel("Semester");
	l7.setForeground(new Color(47, 79, 79));
	l7.setFont(new Font("Tahoma", Font.BOLD, 14));
	l7.setBounds(73, 240, 90, 22);
	contentPane.add(l7);

	t1 = new JTextField();
	t1.setForeground(new Color(47, 79, 79));
	t1.setFont(new Font("Trebuchet MS", Font.BOLD, 14));
	t1.setBounds(169, 54, 198, 20);
	contentPane.add(t1);
	t1.setColumns(10);

	t2 = new JTextField();
	t2.setForeground(new Color(47, 79, 79));
	t2.setFont(new Font("Trebuchet MS", Font.BOLD, 14));
	t2.setColumns(10);
	t2.setBounds(169, 87, 198, 20);
	contentPane.add(t2);

	t3 = new JTextField();
	t3.setForeground(new Color(47, 79, 79));
	t3.setFont(new Font("Trebuchet MS", Font.BOLD, 14));
	t3.setColumns(10);
	t3.setBounds(169, 120, 198, 20);
	contentPane.add(t3);
        
	t4 = new JTextField();
	t4.setForeground(new Color(47, 79, 79));
	t4.setFont(new Font("Trebuchet MS", Font.BOLD, 14));
	t4.setColumns(10);
	t4.setBounds(169, 150, 198, 20);
	contentPane.add(t4);
        
	t5 = new JTextField();
	t5.setForeground(new Color(47, 79, 79));
	t5.setFont(new Font("Trebuchet MS", Font.BOLD, 14));
	t5.setColumns(10);
	t5.setBounds(169, 180, 198, 20);
	contentPane.add(t5);
	
        t6 = new JTextField();
	t6.setForeground(new Color(47, 79, 79));
	t6.setFont(new Font("Trebuchet MS", Font.BOLD, 14));
	t6.setColumns(10);
	t6.setBounds(169, 210, 198, 20);
	contentPane.add(t6);
        
        t7 = new JTextField();
	t7.setForeground(new Color(47, 79, 79));
	t7.setFont(new Font("Trebuchet MS", Font.BOLD, 14));
	t7.setColumns(10);
	t7.setBounds(169, 240, 198, 20);
	contentPane.add(t7);

        b1 = new JButton("Search");
	b1.addActionListener(this);
	b1.setBorder(new CompoundBorder(new LineBorder(new Color(128, 128, 128)), null));
        b1.setFont(new Font("Trebuchet MS", Font.BOLD, 15));
	b1.setBounds(382, 49, 60, 33);
        b1.setBackground(Color.BLACK);
        b1.setForeground(Color.WHITE);
	contentPane.add(b1);

	b_1 = new JButton("Issue");
	b_1.addActionListener(this);
	b_1.setBorder(new CompoundBorder(new LineBorder(new Color(105, 105, 105)), null));
	b_1.setFont(new Font("Trebuchet MS", Font.BOLD, 15));
	b_1.setBounds(152, 330, 108, 33);
        b_1.setBackground(Color.BLACK);
        b_1.setForeground(Color.WHITE);
        contentPane.add(b_1);
                
	b_2 = new JButton("Back");
	b_2.addActionListener(this);
	b_2.setBorder(new CompoundBorder(new LineBorder(new Color(105, 105, 105)), null));
	b_2.setFont(new Font("Trebuchet MS", Font.BOLD, 15));
	b_2.setBounds(312, 330, 108, 33);
        b_2.setBackground(Color.BLACK);
        b_2.setForeground(Color.WHITE);
        contentPane.add(b_2);

        JLabel doi = new JLabel("Date of Issue");
        doi.setBounds(73,270,108,33);
        contentPane.add(doi);
        
        dateChooser = new JDateChooser();
        dateChooser.setBounds(170,270,195,25);
        contentPane.add(dateChooser);

	JPanel panel = new JPanel();
	panel.setBorder(new TitledBorder(new LineBorder(new Color(138, 43, 226), 2), "Student Details", TitledBorder.LEADING,
			TitledBorder.TOP, null, new Color(0, 0, 255)));
	panel.setBounds(10, 29, 458, 344);
	contentPane.add(panel);
        
        panel.setBackground(Color.WHITE);
        contentPane.setBackground(Color.WHITE);
 
// Add Book

	JLabel l_1 = new JLabel("Name");
	l_1.setForeground(new Color(47, 79, 79));
	l_1.setFont(new Font("Tahoma", Font.BOLD, 14));
	l_1.setBounds(498, 84, 90, 22);
	contentPane.add(l_1);

	JLabel l_2 = new JLabel("ISBN");
	l_2.setForeground(new Color(47, 79, 79));
	l_2.setFont(new Font("Tahoma", Font.BOLD, 14));
	l_2.setBounds(498, 117, 90, 22);
	contentPane.add(l_2);

	JLabel l_3 = new JLabel("Publisher");
	l_3.setForeground(new Color(47, 79, 79));
	l_3.setFont(new Font("Tahoma", Font.BOLD, 14));
	l_3.setBounds(498, 150, 90, 22);
	contentPane.add(l_3);

	JLabel l_4 = new JLabel("Price");
	l_4.setForeground(new Color(47, 79, 79));
	l_4.setFont(new Font("Tahoma", Font.BOLD, 14));
	l_4.setBounds(498, 216, 90, 22);
	contentPane.add(l_4);

	JLabel l_5 = new JLabel("Pages");
	l_5.setForeground(new Color(47, 79, 79));
	l_5.setFont(new Font("Tahoma", Font.BOLD, 14));
	l_5.setBounds(498, 249, 90, 22);
	contentPane.add(l_5);

	JLabel l_6 = new JLabel("Book_id");
	l_6.setForeground(new Color(47, 79, 79));
	l_6.setFont(new Font("Tahoma", Font.BOLD, 14));
	l_6.setBounds(498, 51, 90, 22);
	contentPane.add(l_6);

	JLabel l_7 = new JLabel("Edition");
	l_7.setForeground(new Color(47, 79, 79));
	l_7.setFont(new Font("Tahoma", Font.BOLD, 14));
	l_7.setBounds(498, 183, 90, 22);
	contentPane.add(l_7);

	t_1 = new JTextField();
	t_1.setForeground(new Color(47, 79, 79));
	t_1.setFont(new Font("Trebuchet MS", Font.BOLD, 14));
	t_1.setBounds(599, 54, 198, 20);
	contentPane.add(t_1);
	t_1.setColumns(10);

	t_2 = new JTextField();
	t_2.setForeground(new Color(47, 79, 79));
	t_2.setFont(new Font("Trebuchet MS", Font.BOLD, 14));
	t_2.setColumns(10);
	t_2.setBounds(599,87, 198, 20);
	contentPane.add(t_2);

	t_3 = new JTextField();
	t_3.setForeground(new Color(47, 79, 79));
	t_3.setFont(new Font("Trebuchet MS", Font.BOLD, 14));
	t_3.setColumns(10);
	t_3.setBounds(599, 120, 198, 20);
	contentPane.add(t_3);

	t_4 = new JTextField();
	t_4.setForeground(new Color(47, 79, 79));
	t_4.setFont(new Font("Trebuchet MS", Font.BOLD, 14));
	t_4.setColumns(10);
	t_4.setBounds(599, 153, 198, 20);
	contentPane.add(t_4);

        t_5 = new JTextField();
	t_5.setForeground(new Color(47, 79, 79));
	t_5.setFont(new Font("Trebuchet MS", Font.BOLD, 14));
	t_5.setColumns(10);
	t_5.setBounds(599, 219, 198, 20);
	contentPane.add(t_5);

	t_6 = new JTextField();
	t_6.setForeground(new Color(47, 79, 79));
	t_6.setFont(new Font("Trebuchet MS", Font.BOLD, 14));
	t_6.setColumns(10);
	t_6.setBounds(599, 252, 198, 20);
	contentPane.add(t_6);
        
	t_7 = new JTextField();
	t_7.setForeground(new Color(47, 79, 79));
	t_7.setFont(new Font("Trebuchet MS", Font.BOLD, 14));
	t_7.setColumns(10);
	t_7.setBounds(599, 185, 198, 20);
	contentPane.add(t_7);
        
        b_4 = new JButton("Search");
	b_4.addActionListener(this);
	b_4.setBorder(new CompoundBorder(new LineBorder(new Color(128, 128, 128)), null));
        b_4.setFont(new Font("Trebuchet MS", Font.BOLD, 15));
	b_4.setBounds(802, 49, 60, 29);
        b_4.setBackground(Color.BLACK);
        b_4.setForeground(Color.WHITE);
	contentPane.add(b_4);
        
        JPanel panel2 = new JPanel();
	panel2.setBorder(new TitledBorder(new LineBorder(new Color(138, 43, 226), 2), "Issue Book", TitledBorder.LEADING,
			TitledBorder.TOP, null, new Color(0, 0, 255)));
	panel2.setBounds(470, 29, 398, 344);
	contentPane.add(panel2);
        
        panel2.setBackground(Color.WHITE);
        contentPane.setBackground(Color.WHITE);

    }
    public void actionPerformed(ActionEvent ae)
    {  try{    
            conn con = new conn();
            // search book by id
        if (ae.getSource() == b_4)
        {
            String sql = "select * from book where book_id = ?";
            PreparedStatement st = con.c.prepareStatement(sql);
            st.setString(1,t_1.getText() );
            ResultSet rs = st.executeQuery();
            while(rs.next())
            {
                t_2.setText(rs.getString("name"));
                t_3.setText(rs.getString("ISBN"));
                t_4.setText(rs.getString("Publisher"));
                t_5.setText(rs.getString("Edition"));
                t_6.setText(rs.getString("Price"));
                t_7.setText(rs.getString("Pages"));
            }
            rs.close();
            st.close();
                        
          }
        // Search student by id
        if(ae.getSource()== b1){
            String sql = "select * from Student where Student_id = ?";
            PreparedStatement st = con.c.prepareStatement(sql);
            st.setString(1,t1.getText() );
            ResultSet rs = st.executeQuery();
            while(rs.next())
            {
                t2.setText(rs.getString("name"));
                t3.setText(rs.getString("fathers"));
                t4.setText(rs.getString("course"));
                t5.setText(rs.getString("branch"));
                t6.setText(rs.getString("semester"));
            }   
             rs.close();
             st.close();
          }
        // issue button
        if(ae.getSource() == b_1)
        {
            String sql = "Insert into issueBook(book_id,student_id,bname,sname,course,branch,dateofIssue) values(?,?,?,?,?,?,?))";
            PreparedStatement st = con.c.prepareStatement(sql);
            st.setString(1,t_1.getText() );
            st.setString(2,t1.getText() );
            st.setString(3,t_2.getText() );
            st.setString(4,t4.getText() );
            st.setString(5,t5.getText() );
            st.setString(6,t6.getText() );
            st.setString(7,((JTextField) dateChooser.getDateEditor().getUiComponent()).getText());
            int rs = st.executeUpdate();
            if (rs>0){
            JOptionPane.showMessageDialog(null,"Book issued successfully");
             }
            else{JOptionPane.showMessageDialog(null, "error");}
        }      
        // back button
        if(ae.getSource()== b_2){
            this.setVisible(false);
            new Home().setVisible(true);
        }
        con.c.close();
    }
    catch(Exception e){
        e.printStackTrace();
    }}
}


