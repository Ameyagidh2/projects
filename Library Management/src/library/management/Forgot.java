/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package library.management;
import java.awt.*;
import javax.swing.*;
import java.sql.*;
import java.awt.event.*;
import javax.swing.border.*;
/**
 *
 * @author ameya
 */
public class Forgot extends JFrame implements ActionListener{
    
    private JPanel contentPanel;
    private JLabel j1,j2,j3,j4,j5;
    private JTextField t1, t2, t3, t4, t5;
    private JButton b1, b2, b3;
    
    public static void main(String[] args) {
        new Forgot().setVisible(true);
    }
    
    public Forgot(){
        // Name and Security Question have to be filledin by the database itself
        
        setBounds(500,200,650,500);
        contentPanel = new JPanel();// Works as div
        setContentPane(contentPanel);
        contentPanel.setBackground(Color.WHITE);
        contentPanel.setLayout(null);
        
        t1 = new JTextField();
        t1.setForeground(new Color(0,0,0));
        t1.setBounds(277,88,139,20);
        t1.setFont(new Font("Tahoma", Font.BOLD, 12));
        contentPanel.add(t1);
        t1.setColumns(10);
        
        t2 = new JTextField();
        t2.setEditable(false);
        t2.setForeground(new Color(0,0,0));
        t2.setBounds(277,123,139,20);
        t2.setFont(new Font("Tahoma", Font.BOLD, 12));
        contentPanel.add(t2);
        t2.setColumns(10);
        
        t3 = new JTextField();
        t3.setEditable(false);
        t3.setForeground(new Color(0,0,0));
        t3.setForeground(Color.WHITE);
        t3.setBounds(277,161,139,20);
        t3.setFont(new Font("Tahoma", Font.BOLD, 12));
        contentPanel.add(t3);
        t3.setColumns(10);
        
        t4 = new JTextField();
        t4.setForeground(new Color(0,0,0));
        t4.setEditable(false);
        t4.setBounds(277,224,139,20);
        t4.setFont(new Font("Tahoma", Font.BOLD, 12));
        contentPanel.add(t4);
        t4.setColumns(10);
        
        t5 = new JTextField();
        t5.setForeground(new Color(0,0,0));
        t5.setBounds(277,193,139,20);
        t5.setFont(new Font("Tahoma", Font.BOLD, 12));
        contentPanel.add(t5);
        t5.setColumns(10);
        
        
        
        b1 = new JButton("Search");
        b1.addActionListener(this);
        b1.setBackground(Color.BLACK);
        b1.setForeground(Color.WHITE);
        b1.setBounds(428,83,80,29);
        b1.setFont(new Font("Tahoma", Font.BOLD, 12));
        contentPanel.add(b1);
        
        b2 = new JButton("Retrieve");
        b2.addActionListener(this);
        b2.setBackground(Color.BLACK);
        b2.setForeground(Color.WHITE);
        b2.setBounds(428,188,80,29);
        b2.setFont(new Font("Tahoma", Font.BOLD, 10));
        contentPanel.add(b2);
        
        b3 = new JButton("Back");
        b3.addActionListener(this);
        b3.setBackground(Color.BLACK);
        b3.setForeground(Color.WHITE);
        b3.setBounds(233,270,101,29);
        b3.setFont(new Font("Tahoma", Font.BOLD, 12));
        contentPanel.add(b3);
        
        j1 = new JLabel("Username");
        j1.setBounds(109,83,104,21);
        j1.setFont(new Font("Tahoma", Font.BOLD, 12));
        contentPanel.add(j1);
        
        j2 = new JLabel("Name");
        j2.setBounds(109, 122, 104, 21);
        j2.setFont(new Font("Tahoma", Font.BOLD, 12));
        contentPanel.add(j2);

        
        j3 = new JLabel("Your Security Question");
        j3.setBounds(109, 154, 154, 21);
        j3.setFont(new Font("Tahoma", Font.BOLD, 12));
        contentPanel.add(j3);
        
        j4 = new JLabel("Answer:");
        j4.setBounds(109, 188, 104, 21);
        j4.setFont(new Font("Tahoma", Font.BOLD, 12));
        contentPanel.add(j4);
        
        j5 = new JLabel("Password");
        j5.setBounds(109, 224, 104, 21);
        j5.setFont(new Font("Tahoma", Font.BOLD, 12));
        contentPanel.add(j5);
        
        JPanel panel = new JPanel();
        panel.setBorder(new TitledBorder(new LineBorder(new Color(128,128,0),2),"Forgot-Password",
                TitledBorder.LEADING,TitledBorder.TOP,null, new Color(34,129,34)));
        // leading top is top left
        panel.setBounds(31, 46, 550, 300);
        panel.setBackground(Color.WHITE);
        contentPanel.add(panel);      
        
    }
    public void actionPerformed(ActionEvent ae){
        try{
            conn con = new conn();
            if (ae.getSource() == b1)
            {
                String sql = "select * from account where username = ?";
                PreparedStatement st = con.c.prepareStatement(sql);
                
                st.setString(1, t1.getText());// get text to get the username 
                
                // ResultSet is used to get entire table
                
                ResultSet rs = st.executeQuery();
                
                while(rs.next())
                    // if match of a particular username is reached
                {
                    // get next rows in the data table
                   // get String to loop through each column in the database table
                    t2.setText(rs.getString("name"));
                    t3.setText(rs.getString("sec_q"));                    
                    
                }
            }
                
                if (ae.getSource() == b2)
                {
                    // get password if the entered security answer matches with the database answer
                   String sql = "select * from account where sec_ans = ?";
                   PreparedStatement st = con.c.prepareStatement(sql);
                   
                   st.setString(1,t4.getText());
                   
                   ResultSet rs = st.executeQuery();
                   while(rs.next()){
                   // match the sec_ans
                   t5.setText(rs.getString("password"));
                   // get the password retrieved
                   
                   }
                }   
                if (ae.getSource() == b3)
                {
                    this.setVisible(false);
                    new Login_user().setVisible(true);
                }
                
                              
            }
            
                    
         catch(Exception e)
        {  
         
        }
        
    
    }
    
    
    
}
