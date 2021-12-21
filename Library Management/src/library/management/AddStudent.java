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

public class AddStudent extends JFrame implements ActionListener
{
    
    private JPanel contentPane;
    private JTextField t1,t2,t3;
    private JButton b1,b2;
    JComboBox comboBox1,comboBox2,comboBox3,comboBox4;
        
    public static void main(String[] args) {
	new AddStudent().setVisible(true);
    }

    public void random() {
        Random rd = new Random();
	t1.setText("" + rd.nextInt(1000 + 1));
    }

    public AddStudent() {
        setBounds(600, 200, 518, 442);
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
	l5.setBounds(73, 191, 90, 22);
	contentPane.add(l5);

	JLabel l6 = new JLabel("Year");
	l6.setForeground(new Color(47, 79, 79));
	l6.setFont(new Font("Tahoma", Font.BOLD, 14));
	l6.setBounds(73, 221, 90, 22);
	contentPane.add(l6);

	JLabel l7 = new JLabel("Semester");
	l7.setForeground(new Color(47, 79, 79));
	l7.setFont(new Font("Tahoma", Font.BOLD, 14));
	l7.setBounds(73, 243, 90, 22);
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

	comboBox1 = new JComboBox();
	comboBox1.setModel(new DefaultComboBoxModel(new String[] { "1", "2", "3", "4", "5", "6", "7", "8", "9" }));
	comboBox1.setBounds(169, 153, 194, 20);
	contentPane.add(comboBox1);
	
        comboBox2 = new JComboBox();
	comboBox2.setModel(new DefaultComboBoxModel(new String[] { "1", "2", "3", "4", "5", "6", "7", "8", "9" }));
	comboBox2.setBounds(169, 190, 194, 20);
	contentPane.add(comboBox2);
	

	comboBox3 = new JComboBox();
	comboBox3.setModel(new DefaultComboBoxModel(new String[] { "1", "2", "3", "4", "5", "6", "7", "8", "9" }));
	comboBox3.setBounds(169, 220, 194, 20);
	contentPane.add(comboBox3);

	comboBox4 = new JComboBox();
	comboBox4.setModel(new DefaultComboBoxModel(new String[] { "1", "2", "3", "4", "5", "6", "7", "8", "9" }));
	comboBox4.setBounds(169, 246, 194, 20);
	contentPane.add(comboBox4);

        b1 = new JButton("Add");
	b1.addActionListener(this);
	b1.setBorder(new CompoundBorder(new LineBorder(new Color(128, 128, 128)), null));
        b1.setFont(new Font("Trebuchet MS", Font.BOLD, 15));
	b1.setBounds(102, 300, 100, 33);
        b1.setBackground(Color.BLACK);
        b1.setForeground(Color.WHITE);
	contentPane.add(b1);

	b2 = new JButton("Back");
	b2.addActionListener(this);
	b2.setBorder(new CompoundBorder(new LineBorder(new Color(105, 105, 105)), null));
	b2.setFont(new Font("Trebuchet MS", Font.BOLD, 15));
	b2.setBounds(212, 300, 108, 33);
        b2.setBackground(Color.BLACK);
        b2.setForeground(Color.WHITE);
	
	contentPane.add(b2);

	JPanel panel = new JPanel();
	panel.setBorder(new TitledBorder(new LineBorder(new Color(138, 43, 226), 2), "Add Student", TitledBorder.LEADING,
			TitledBorder.TOP, null, new Color(0, 0, 255)));
	panel.setBounds(10, 29, 398, 344);
	contentPane.add(panel);
        
        panel.setBackground(Color.WHITE);
        contentPane.setBackground(Color.WHITE);
	random();
    }
    public void actionPerformed(ActionEvent ae)
    {  try{    
        conn con = new conn();
        if (ae.getSource() == b1)
        {
            String sql = "Insert into Student(Student_id name,father,course, branch,year,semester) values(?,?,?,?,?,?,?)";
            PreparedStatement st  = con.c.prepareStatement(sql);
            
            st.setString(1,t1.getText() );
            st.setString(1,t2.getText() );
            st.setString(1,t3.getText() );
            st.setString(1,(String)comboBox1.getSelectedItem());
            st.setString(1,(String)comboBox2.getSelectedItem());
            st.setString(1,(String)comboBox3.getSelectedItem());
            st.setString(1,(String)comboBox4.getSelectedItem());
            
            int rs = st.executeUpdate();
            if (rs > 0)
            {
            JOptionPane.showMessageDialog(null,"Successfully Added");
            }
            else
            {
                JOptionPane.showMessageDialog(null,"Error");
                t1.setText("");
                t2.setText("");
                t3.setText("");
                st.close();
                    }
            
          }
        if(ae.getSource()== b2){
        this.setVisible(false);
        new Home().setVisible(true);
        }
        con.c.close();
    }
    catch(Exception e){
        
    }}
}
