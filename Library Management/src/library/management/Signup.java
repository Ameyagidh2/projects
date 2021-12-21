/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package library.management;

/**
 *
 * @author ameya
 */

// in set content pane add controlpanel in which add items

import java.awt.*;
import javax.swing.*;
import java.sql.*;
import java.awt.event.*;
import javax.swing.border.*;

public class Signup extends JFrame implements ActionListener{
    private  JPanel controlPanel;
    private  JTextField  j1;
    private  JTextField  j2;
    private  JTextField  j3;
    private  JTextField  j4;
    
    private JLabel l1;
    private JLabel l2;
    private JLabel l3;
    private JLabel l4;
    
    private JComboBox combobox;
    
    private JButton b1, b2;
    
    public static void main(String[] args)
    {
        new Signup().setVisible(true);
    }
    public Signup()
    {   setBounds(600, 250,606, 406);
        controlPanel = new JPanel();
        //controlPanel.setForeground(new Color(34, 139, 34));
	controlPanel.setBorder(new EmptyBorder(5, 5, 5, 5));
        controlPanel.setBackground(Color.WHITE);
        setContentPane(controlPanel);
        controlPanel.setLayout(null);
    
        j1 = new JTextField();
        j1.setColumns(10);
        j1.setBounds(265, 91, 148, 20);
        controlPanel.add(j1);
        
        j2 = new JTextField();
        j2.setColumns(10);
        j2.setBounds(265, 128, 148, 20);
        controlPanel.add(j2);
        
        j3 = new JTextField();
        j3.setColumns(10);
        j3.setBounds(265, 165, 148, 20);
        controlPanel.add(j3);
                
        j4 = new JTextField();
        j4.setColumns(10);
        j4.setBounds(265, 239, 148, 20);
        controlPanel.add(j4);
        
        JLabel l1 = new JLabel("Username: ");
        l1.setBackground(Color.WHITE);
        l1.setForeground(Color.BLACK);
        l1.setFont(new Font("Tahoma",Font.BOLD, 14));
        l1.setBounds(99, 86, 92, 26);
        controlPanel.add(l1);
        
        JLabel l2 = new JLabel("Name: ");
        l2.setBackground(Color.WHITE);
        l2.setForeground(Color.BLACK);
        l2.setBounds(99, 123, 92, 26);
        controlPanel.add(l2);
        
        
        JLabel l3 = new JLabel("Password: ");
        l3.setBackground(Color.WHITE);
        l3.setForeground(Color.BLACK);
        l3.setFont(new Font("Tahoma",Font.BOLD, 14));
        l3.setBounds(99, 160, 92, 26);
        controlPanel.add(l3);
        
        JLabel l4 = new JLabel("Answer: ");
        l4.setBackground(Color.WHITE);
        l4.setForeground(Color.BLACK);
        l4.setFont(new Font("Tahoma",Font.BOLD, 14));
        l4.setBounds(99, 234, 140, 26);
        controlPanel.add(l4);
        
        b1 = new JButton("Create");
        b1.addActionListener(this);
        b1.setFont(new Font("Tahoma", Font.BOLD, 13));
	b1.setBounds(140, 289, 100, 30);
        b1.setBackground(Color.WHITE);
        b1.setForeground(Color.BLACK);
        controlPanel.add(b1);
        
        b2 = new JButton("Back");
        b2.addActionListener(this);
        b2.setFont(new Font("Tahoma", Font.BOLD, 13));
	b2.setBounds(300, 289, 100, 30);
        b2.setBackground(Color.WHITE);
        b2.setForeground(Color.BLACK);
        controlPanel.add(b2);
        
        JComboBox comboBox = new JComboBox();
        comboBox.setModel(new DefaultComboBoxModel(new String[] { "Your NickName?", "Your Lucky Number?",
			"Your child SuperHero?", "Your childhood Name ?" }));
        comboBox.setBounds(265, 202, 148, 20);
        controlPanel.add(comboBox);
        
        JLabel l5 = new JLabel("Security Question: ");
        l5.setBackground(Color.WHITE);
        l5.setForeground(Color.BLACK);
        l5.setFont(new Font("Tahoma",Font.BOLD, 14));
        l5.setBounds(99, 194, 140, 26);
        controlPanel.add(l5);
        
        // Thi is the border j panel
        JPanel panel = new JPanel();
        panel.setBorder(new TitledBorder(new LineBorder(new Color(128,128,0),2),"Create - Account",
                TitledBorder.LEADING,TitledBorder.TOP,null, new Color(34,129,34)));
        // leading top is top left
        panel.setBounds(31, 46, 476, 296);
        panel.setBackground(Color.WHITE);
        controlPanel.add(panel);
        
    }
        public void actionPerformed(ActionEvent ae){              
        // question mark for Prepare Statement
        try
        {   conn con = new conn();
            if(ae.getSource() == b1)
          {
            // create button is clicked
            String sql = "insert into account(username, name, password, sec_key,sec_ans) values(?,?,?,?)";
            // for insert use Update select use execute
            PreparedStatement st = con.c.prepareStatement(sql);
            // c is object of connection
            st.setString(1, j1.getText());
            st.setString(2, j2.getText());
            st.setString(3, j3.getText());
            st.setString(4, (String)combobox.getSelectedItem());
// combobox returns an object hence needs to be type casted to string
            st.setString(5, j4.getText());
            
            int i = st.executeUpdate(sql);
            // i if try
            if (i > 0)
            {
            // i is the greater value
                JOptionPane.showMessageDialog(null, "Added the user to database");
            }
            // set the fields to blank again
            
            j1.setText("");
            j2.setText("");
            j3.setText("");
            j4.setText("");
        }   
        if (ae.getSource() == b2){
            // back button clicked
//             System.out.println("li");
            this.setVisible(false);
            // go back to login screen
            new Login_user().setVisible(true);
//            System.out.println("li22");
        }
        }catch(Exception e)
        {
//            System.out.println("Error creating a New user");
        }
               
        }   
        
    }






