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

public class Home extends JFrame implements ActionListener{
    
    private JLabel l1,l2,l3,l4,l5;
    private JTextField j1,j2,j3,j4,j5;
    private JPanel controlPanel;
    private JButton b1,b2,b3,b4,b5,b6;
                
    public Home()
    {   
        controlPanel = new JPanel();
        setBounds(400, 150,1000, 800);
        setContentPane(controlPanel);
        controlPanel.setLayout(null);
         
        // Menu Bar has menus which have items
        
        // Menubar
        JMenuBar menubar = new JMenuBar();
        menubar.setBounds(0,10,1000,35);
        menubar.setBackground(Color.CYAN);
        controlPanel.add(menubar);
        
        JMenu Record_menu = new JMenu("Record");
        Record_menu.setBackground(Color.WHITE);
        Record_menu.setForeground(Color.BLACK);
        menubar.add(Record_menu);
        
        JMenu help_menu = new JMenu("Help");
        help_menu.setBackground(Color.WHITE);
        help_menu.setForeground(Color.BLACK);
        menubar.add(help_menu);
        
        JMenu exit_menu = new JMenu("Exit");
        exit_menu.setBackground(Color.WHITE);
        exit_menu.setForeground(Color.BLACK);
        menubar.add(exit_menu);
          
        JMenuItem Logout = new JMenuItem("Logout");
        Logout.addActionListener(this);
        Logout.setBackground(Color.WHITE);
        Logout.setForeground(Color.BLACK);
        exit_menu.add(Logout);
        
        JMenuItem Exit = new JMenuItem("Exit");
        Exit.addActionListener(this);
        Exit.setBackground(Color.WHITE);
        Exit.setForeground(Color.BLACK);
        exit_menu.add(Exit);
        
        JMenuItem book_details = new JMenuItem("Book Details");
        book_details.addActionListener(this);
        book_details.setBackground(Color.WHITE);
        book_details.setForeground(Color.BLACK);
        Record_menu.add(book_details);
        
        JMenuItem Student_details = new JMenuItem("Student Details");
        Student_details.addActionListener(this);
        Student_details.setBackground(Color.WHITE);
        Student_details.setForeground(Color.BLACK);
        Record_menu.add(Student_details);
        
        JMenuItem Read_Me = new JMenuItem("Read Me");
        Read_Me.addActionListener(this);
        Read_Me.setBackground(Color.WHITE);
        Read_Me.setForeground(Color.BLACK);
        help_menu.add(Read_Me);
        
        JMenuItem About_us = new JMenuItem("About Us");
        About_us.addActionListener(this);
        About_us.setBackground(Color.WHITE);
        About_us.setForeground(Color.BLACK);
        help_menu.add(About_us);
     
        JLabel  l2 = new JLabel("");
        l2.setVerticalAlignment(SwingConstants.TOP);
        // we need to add image to a label then again convert it back to an image
        ImageIcon i1 = new ImageIcon(ClassLoader.getSystemResource("library/management/icons/second.png"));  
        // now this is the image icon
        Image i2 = i1.getImage().getScaledInstance(150, 150, Image.SCALE_DEFAULT);
        ImageIcon i3 = new ImageIcon(i2);
        l2  = new JLabel(i3);
        l2.setBounds(160, 140, 159, 152);
        controlPanel.add(l2);
        
        JLabel  l3 = new JLabel("");
        l3.setVerticalAlignment(SwingConstants.TOP);
        // we need to add image to a label then again convert it back to an image
        ImageIcon i4 = new ImageIcon(ClassLoader.getSystemResource("library/management/icons/third.png"));  
        // now this is the image icon
        Image i5 = i4.getImage().getScaledInstance(200, 200, Image.SCALE_DEFAULT);
        ImageIcon i6 = new ImageIcon(i5);
        l3  = new JLabel(i6);
        l3.setBounds(370, 140, 159, 152);
        controlPanel.add(l3);
        
        JLabel  l1 = new JLabel("");
        l1.setVerticalAlignment(SwingConstants.TOP);
        // we need to add image to a label then again convert it back to an image
        ImageIcon i7 = new ImageIcon(ClassLoader.getSystemResource("library/management/icons/fourth.png"));  
        // now this is the image icon
        Image i8 = i7.getImage().getScaledInstance(150, 150, Image.SCALE_DEFAULT);
        ImageIcon i9 = new ImageIcon(i8);
        l1 = new JLabel(i9);
        l1.setBounds(570, 140, 159, 152);
        controlPanel.add(l1);
        
        JLabel l4 = new JLabel("");
        ImageIcon i10 = new ImageIcon(ClassLoader.getSystemResource("library/management/icons/fifth.png"));
        Image i11 = i10.getImage().getScaledInstance(150,150, Image.SCALE_DEFAULT);
        ImageIcon i12 = new ImageIcon(i11);
        // converting image icon back to label formm
        l4 = new JLabel(i12);
        l4.setBounds(160, 420, 159, 152);
        controlPanel.add(l4);
        
        JLabel l5 = new JLabel("");
        ImageIcon i13 = new ImageIcon(ClassLoader.getSystemResource("library/management/icons/sixth.png"));
        Image i14 = i13.getImage().getScaledInstance(150,150,Image.SCALE_DEFAULT);
        ImageIcon i15 = new ImageIcon(i14);
        l5 = new JLabel(i15);
        l5.setBounds(360,420,159,152);
        controlPanel.add(l5);
        
        JLabel l6 = new JLabel("");
        ImageIcon i16 = new ImageIcon(ClassLoader.getSystemResource("library/management/icons/seventh.png"));
        Image i17 = i16.getImage().getScaledInstance(150,150,Image.SCALE_DEFAULT);
        ImageIcon i18 = new ImageIcon(i17);
        l6 = new JLabel(i18);
        l6.setBounds(560,420,159,152);
        controlPanel.add(l6);
        
        JPanel j1 = new JPanel();
        j1.setBorder(new TitledBorder(new LineBorder(new Color(128,128,0),2),"Operation",
                TitledBorder.LEADING,TitledBorder.TOP,null,new Color(233,150,122)));
        j1.setBounds(60,100,900,270);
        controlPanel.add(j1);
        
        JPanel j2 = new JPanel();
        j2.setBorder(new TitledBorder(new LineBorder(new Color(128,128,0),2),"  Action",
                TitledBorder.LEADING,TitledBorder.TOP,null,new Color(233,150,122)));
        j2.setBounds(60,390,900,270);
        controlPanel.add(j2);
        
        
        b1 = new  JButton("Add Book");
        b1.addActionListener(this);
        b1.setBackground(Color.BLACK);
        b1.setForeground(Color.WHITE);
        b1.setBounds(160,300,159,40);
        controlPanel.add(b1);
        
        b2 = new  JButton("Statistics");
        b2.addActionListener(this);
        b2.setBackground(Color.BLACK);
        b2.setForeground(Color.WHITE);
        b2.setBounds(370,300,159,40);
        controlPanel.add(b2);
        
        b3 = new  JButton("Add Student");
        b3.addActionListener(this);
        b3.setBackground(Color.BLACK);
        b3.setForeground(Color.WHITE);
        b3.setBounds(560,300,159,40);
        controlPanel.add(b3);
        
        b4 = new  JButton("Issue Book");
        b4.addActionListener(this);
        b4.setBackground(Color.BLACK);
        b4.setForeground(Color.WHITE);
        b4.setBounds(160,600,159,40);
        controlPanel.add(b4);
        
        b5 = new  JButton("Return Book");
        b5.addActionListener(this);
        b5.setBackground(Color.BLACK);
        b5.setForeground(Color.WHITE);
        b5.setBounds(370,600,159,40);
        controlPanel.add(b5);
        
        b6 = new  JButton("About Us");
        b6.addActionListener(this);
        b6.setBackground(Color.BLACK);
        b6.setForeground(Color.WHITE);
        b6.setBounds(560,600,159,40);
        controlPanel.add(b6);
    }
    
    public void  actionPerformed(ActionEvent ae)
    {
        String msg = ae.getActionCommand();
        if(msg.equals("Logout"))
        {
        // getActionCommand() for menubar
            setVisible(false);
            new Login_user().setVisible(true);
        }
        else if(msg.equals("Exit"))
        {
        System.exit(0);//ABORT
        }
        else if(msg.equals("Read Me"))
        {
        new ReadMe().setVisible(true);
        }   
        
        else if(msg.equals("About Us"))
        {
        new AboutUs().setVisible(true);
        }     
        
        else if(msg.equals("Book Details"))
        {   
            setVisible(false);
            new BookDetails().setVisible(true);
        }       
        
        else if(msg.equals("Student Details"))
        {
            setVisible(false);
            new StudentDetails().setVisible(true);
        }       
        
        
        
        
       if(ae.getSource() == b1)
        {
            this.setVisible(false);
            new AddBook().setVisible(true);
        }
        if(ae.getSource() == b2)
        {
            this.setVisible(false);
            new Statistics().setVisible(true);
         }
        if(ae.getSource() == b3)
        {
            this.setVisible(false);
            new AddStudent().setVisible(true);
         }
        if(ae.getSource() == b4)
        {
            this.setVisible(false);
            new IssueBook().setVisible(true);
         }
        if(ae.getSource() == b5)
        {
            this.setVisible(false);
            new ReturnBook().setVisible(true);
         }
        if(ae.getSource() == b6)
        {
            this.setVisible(false);
            new AboutUs().setVisible(true);
         }
    }
        
    public static void main(String[] args) 
    {
        new Home().setVisible(true);
    }
 
}
